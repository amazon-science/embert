import glob
import json
import os
import statistics
from datetime import datetime

import cv2
import numpy as np
import torch

from eval import Eval
from grolp import AlfredPredictor
from grolp.envs.thor_env import ThorEnv
from grolp.gen import constants
from grolp.utils.video_util import VideoSaver
from scripts.generate_maskrcnn import CustomImageLoader, MaskRCNNDetector


def get_image_index(save_path):
    return len(glob.glob(save_path + '/*.png'))


classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp',
                                       'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


def save_image(image_to_save, save_path):
    # rgb
    rgb_save_path = save_path
    rgb_image = image_to_save

    # dump images
    im_ind = get_image_index(rgb_save_path)
    cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image)
    return im_ind


class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, split_id, predictor, image_loader, region_detector, task_queue, args, lock,
            successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv(player_screen_width=args.frame_size, player_screen_height=args.frame_size)

        try:
            while True:
                if task_queue.qsize() == 0:
                    break

                task = task_queue.get()

                try:
                    traj = predictor.load_task_json(split_id, task)
                    r_idx = task['repeat_idx']
                    print("Evaluating: %s" % (traj['root']))
                    print("No. of trajectories left: %d" % (task_queue.qsize()))
                    cls.evaluate(env, predictor, r_idx, image_loader, region_detector, traj,
                                 args,
                                 lock, successes, failures, results)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Error: " + repr(e))
        except KeyboardInterrupt:
            print("CTRL-C pressed. Stopping...")
        finally:
            # stop THOR
            env.stop()

    @classmethod
    @torch.no_grad()
    def evaluate(cls, env: ThorEnv, predictor: AlfredPredictor, r_idx: int, image_loader: CustomImageLoader,
                 region_detector: MaskRCNNDetector, traj_data, args, lock, successes, failures, results):
        action_mems = None
        state_mems = None
        language_features = None
        language_masks = None
        prev_actions = None
        prev_objects = None

        instruction_idx = 0 if predictor.is_split_model else None
        if args.save_video_path is not None:
            high_res_images_dir = os.path.join(
                os.path.dirname(args.model_path),
                args.save_video_path,
                traj_data["split"],
                traj_data["task_type"],
                traj_data["task_id"],
                str(traj_data["repeat_idx"])
            )
            if not os.path.exists(high_res_images_dir):
                os.makedirs(high_res_images_dir)
        else:
            high_res_images_dir = None

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        high_descs = traj_data['turk_annotations']['anns'][r_idx]["high_descs"]

        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        action_stats = {}

        prev_action = "Start"
        pred_trajectory = []
        prev_image = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']

        if high_res_images_dir is not None:
            image_to_save = env.last_event.frame[:, :, ::-1]
            save_image(image_to_save, high_res_images_dir)

        # used to determine whether we successfully execute an action
        is_action_ok = True
        pred_action = None
        pred_mask = None
        output_dict = None
        action_mask = None
        object_mask = None

        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # check whether previous action was successful
            if is_action_ok:
                # in this situation we just want to execute the current action
                object_features = cls.get_visual_features(
                    env,
                    image_loader,
                    region_detector,
                    args,
                    predictor.cuda_device
                )
                num_objects_in_front = object_features[0]["num_objects"]
                # reinitialise the masks for action replay
                # when a navigation action fails we restrict the
                action_mask, object_mask = predictor.init_masks(num_objects_in_front)

                if args.debug and instruction_idx < len(high_descs):
                    print(f"Instruction ({instruction_idx}/{len(high_descs)}): {high_descs[instruction_idx]}")
                instance = predictor.featurize(traj_data, object_features, instruction_idx)

                # forward model
                output_dict = predictor.predict_instance(
                    instance,
                    action_mems=action_mems,
                    state_mems=state_mems,
                    language_features=language_features,
                    language_masks=language_masks,
                    prev_actions=prev_actions,
                    prev_objects=prev_objects,
                    num_objects_in_front=num_objects_in_front
                )[0]

                pred_action = output_dict["actions"]
                pred_mask = output_dict["masks"]
                action_mems = output_dict["action_mems"]
                state_mems = output_dict["state_mems"]
                prev_objects = output_dict.get("prev_objects")
                prev_actions = output_dict["prev_actions"].unsqueeze(0)

                if prev_objects is not None:
                    prev_objects = prev_objects.unsqueeze(0)

                goto_next_instruction = output_dict["goto_next_instruction"]

                if goto_next_instruction and predictor.is_split_model:
                    # we got to the next instruction now if we predict 'go next'

                    if instruction_idx < len(high_descs):
                        # we increase the index only if we can to avoid out-of-boundaries errors
                        instruction_idx += 1

                        # we also reset the language features so that the model will recompute them
                        language_masks = None
                        language_features = None
                else:
                    # otherwise we reuse the cached language features
                    language_features = output_dict["language_features"].unsqueeze(0)
                    language_masks = output_dict["language_masks"].unsqueeze(0)
            else:
                # in this situation we need to fix something that happened in the previous step
                # we don't need to recompute the visual features here because they are going to be the same

                if pred_action in nav_actions:
                    # current pred action failed. We inspect the probability distribution and we get the next available one
                    pred_action_idx = predictor._model.vocab.get_token_index(pred_action, "low_action_labels")
                    action_mask[pred_action_idx] = 0

                    # we compute the softmax and mask out the already used actions
                    pred_action_idx = np.argmax(output_dict["action_probs"] * action_mask, -1)
                    pred_action = predictor._model.vocab.get_token_from_index(pred_action_idx, "low_action_labels")
                    prev_actions = output_dict["prev_actions"].unsqueeze(0)
                    prev_actions.fill_(pred_action_idx)
                else:
                    pred_object = output_dict["pred_objects"]
                    object_mask[pred_object] = 0

                    object_logits = output_dict["object_probs"]

                    pred_object = np.argmax(object_logits * object_mask, -1)
                    # then we extract the masks for the front view
                    object_masks = output_dict["interactive_object_masks"][0]
                    pred_mask = object_masks[pred_object].squeeze(0)
                    output_dict["pred_objects"] = pred_object
                    prev_objects = output_dict.get("prev_objects")
                    if prev_objects is not None:
                        prev_objects.fill_(pred_object)

            pred_trajectory.append(pred_action)

            # check if <<stop>> was predicted
            if pred_action == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            # print action
            if args.debug:
                print(pred_action)

            # use predicted action and mask (if available) to interact with the env
            is_action_ok, _, _, err, _ = env.va_interact(pred_action, interact_mask=pred_mask,
                                                         smooth_nav=args.smooth_nav,
                                                         debug=args.debug)

            if pred_action not in action_stats:
                action_stats[pred_action] = dict(
                    total=0,
                    success=0
                )

            action_stats[pred_action]['total'] += 1
            if not is_action_ok:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break
            else:
                action_stats[pred_action]['success'] += 1

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

            if high_res_images_dir is not None:
                image_to_save = env.last_event.frame[:, :, ::-1]
                save_image(image_to_save, high_res_images_dir)

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward),
                     'action_stats': action_stats,
                     'api_failure_limit': fails >= args.max_fails,
                     'pred_traj_length': len(pred_trajectory),
                     'pred_trajectory': ','.join(pred_trajectory)
                     }

        if 'plan' in traj_data:
            log_entry['gold_traj_length'] = len(traj_data["plan"]["low_actions"])
            log_entry['gold_trajectory'] = ','.join(
                a["discrete_action"]["action"] for a in traj_data["plan"]["low_actions"])

        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        if high_res_images_dir is not None:
            saver = VideoSaver()
            saver.save(high_res_images_dir, os.path.join(high_res_images_dir, "video.mp4"))

        print("Action prediction analysis")
        for a_type, a_stats in results['all']['action_stats'].items():
            print(
                f"- {a_type} execution success rate = {a_stats['success_rate']:.3f} ({a_stats['success']}/{a_stats['total']})")

        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                    sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                                sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        api_failure_rate = float(sum([1 for entry in successes if entry["api_failure_limit"]]) + \
                                 sum([1 for entry in failures if entry["api_failure_limit"]])) / num_evals

        avg_pred_length = statistics.mean(
            [entry["pred_traj_length"] for entry in successes] +
            [entry["pred_traj_length"] for entry in failures]
        )

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                         'total_goal_conditions': total_goal_conditions,
                                         'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc
        res['avg_pred_traj_length'] = avg_pred_length
        res['avg_api_failure_rate'] = api_failure_rate

        action_stats = dict()

        for s in successes:
            for a_type, stats in s['action_stats'].items():
                if a_type not in action_stats:
                    action_stats[a_type] = {
                        'total': 0,
                        'success': 0
                    }

                action_stats[a_type]['total'] += stats['total']
                action_stats[a_type]['success'] += stats['success']

        for f in failures:
            for a_type, stats in f['action_stats'].items():
                if a_type not in action_stats:
                    action_stats[a_type] = {
                        'total': 0,
                        'success': 0
                    }

                action_stats[a_type]['total'] += stats['total']
                action_stats[a_type]['success'] += stats['success']

        for a_type, stats in action_stats.items():
            action_stats[a_type]["success_rate"] = float(stats['success'] / stats['total'])

        res['action_stats'] = action_stats

        return res

    def create_stats(self):
        '''
        storage for success, failure, and results info
        '''
        self.successes, self.failures = self.manager.list(), self.manager.list()
        self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        results['model_checkpoint'] = os.path.basename(self.args.model_path)
        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

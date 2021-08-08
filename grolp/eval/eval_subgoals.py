import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch

from eval import Eval
from grolp import AlfredPredictor
from grolp.envs.thor_env import ThorEnv
from grolp.gen import constants
from scripts.generate_maskrcnn import CustomImageLoader, MaskRCNNDetector

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp',
                                       'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


class EvalSubgoals(Eval):
    '''
    evaluate subgoals by teacher-forching expert demonstrations
    '''

    # subgoal types
    ALL_SUBGOALS = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject',
                    'SliceObject', 'ToggleObject']

    @classmethod
    def run(cls, split_id, predictor, image_loader, region_detector, task_queue, args, lock,
            successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        # make subgoals list
        subgoals_to_evaluate = cls.ALL_SUBGOALS if args.subgoals.lower() == "all" else args.subgoals.split(',')
        subgoals_to_evaluate = [sg for sg in subgoals_to_evaluate if sg in cls.ALL_SUBGOALS]
        print("Subgoals to evaluate: %s" % str(subgoals_to_evaluate))

        # create empty stats per subgoal
        for sg in subgoals_to_evaluate:
            successes[sg] = list()
            failures[sg] = list()

        try:
            while True:
                if task_queue.qsize() == 0:
                    break

                task = task_queue.get()

                try:
                    traj = predictor.load_task_json(split_id, task)
                    r_idx = task['repeat_idx']
                    subgoal_idxs = [sg['high_idx'] for sg in traj['plan']['high_pddl'] if
                                    sg['discrete_action']['action'] in subgoals_to_evaluate]

                    # we run the model through the entire trajectory here
                    gold_feats = predictor.featurize(traj)
                    gold_forward_out = predictor.predict_instance(gold_feats, is_gold_trajectory=True)

                    for eval_idx in subgoal_idxs:
                        print("No. of trajectories left: %d" % (task_queue.qsize()))
                        cls.evaluate(
                            env, predictor, gold_forward_out, eval_idx,
                            r_idx, image_loader, region_detector, traj,
                            args,
                            lock, successes, failures, results
                        )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Error: " + repr(e))
        finally:
            # stop THOR
            env.stop()

    @classmethod
    def evaluate(cls,
                 env: ThorEnv, predictor: AlfredPredictor, gold_forward_out: Dict[str, Any],
                 eval_idx: int, r_idx: int, image_loader: CustomImageLoader,
                 region_detector: MaskRCNNDetector, traj_data, args, lock, successes, failures, results):
        action_mems = None
        state_mems = None
        language_features = None
        language_masks = None
        prev_actions = None
        prev_objects = None

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # expert demonstration to reach eval_idx-1
        expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if
                               a['high_idx'] < eval_idx]

        # subgoal info
        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
        subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]

        # print subgoal info
        print(
            "Evaluating: %s\nSubgoal %s (%d)\nInstr: %s" % (traj_data['root'], subgoal_action, eval_idx, subgoal_instr))

        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        # extract language features
        # feat = model.featurize([(traj_data, False)], load_mask=False)
        # TODO: load gold trajectory data for expert unrolling

        done, subgoal_success = False, False
        fails = 0
        t = 0
        reward = 0
        is_action_ok = True
        pred_action = None
        pred_mask = None
        output_dict = None
        action_mask = None
        object_mask = None
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        high_descs = traj_data['turk_annotations']['anns'][r_idx]["high_descs"]

        while not done:
            # break if max_steps reached
            if t >= args.max_steps + len(expert_init_actions):
                break

            # expert teacher-forcing upto subgoal
            if t < len(expert_init_actions):
                # get expert action
                action = expert_init_actions[t]
                subgoal_completed = traj_data['plan']['low_actions'][t + 1]['high_idx'] != \
                                    traj_data['plan']['low_actions'][t]['high_idx']
                compressed_mask = action['args']['mask'] if 'mask' in action['args'] else None
                mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None

                # execute expert action
                success, _, _, err, _ = env.va_interact(action['action'], interact_mask=mask,
                                                        smooth_nav=args.smooth_nav, debug=args.debug)
                if not success:
                    print("expert initialization failed")
                    break

                # update transition reward
                _, _ = env.get_transition_reward()
                prev_actions = torch.tensor(
                    [[predictor._model.vocab.get_token_index(action['action'], "low_action_labels")]],
                    dtype=torch.long)
            # subgoal evaluation
            else:
                # TODO: run model with gold sub-trajectory and get memories
                # we do this only the first time though
                if t == len(expert_init_actions):
                    # at this time step we have completely unrolled all the expert demonstrations
                    # so we run the model with the entire gold trajectory up to the current subgoal
                    instruction_idx = eval_idx if predictor.is_split_model else None
                    if t != 0:
                        # in this case we want to initialise the previous memories and make sure that we shift them
                        # accordingly
                        action_mems = []
                        state_mems = []

                        for am in gold_forward_out["action_mems"]:
                            remove_steps = gold_forward_out["trajectory_len"] - len(expert_init_actions)
                            curr_am = am[:-remove_steps].clone()
                            shape = (remove_steps,) + curr_am.shape[1:]
                            curr_am = torch.cat([curr_am.new_zeros(*shape), curr_am])
                            action_mems.append(curr_am)
                        for sm in gold_forward_out["state_mems"]:
                            remove_steps = gold_forward_out["trajectory_len"] - len(expert_init_actions)
                            curr_sm = sm[:-remove_steps].clone()
                            shape = (remove_steps,) + curr_sm.shape[1:]
                            curr_sm = torch.cat([curr_sm.new_zeros(*shape), curr_sm])
                            state_mems.append(curr_sm)

                # after this we run the model normally
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

                    # if args.debug and instruction_idx < len(high_descs):
                    #    print(f"Instruction ({instruction_idx}/{len(high_descs)}): {high_descs[instruction_idx]}")
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
                    prev_actions = output_dict.get("prev_actions")

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

                prev_actions = output_dict["prev_actions"].unsqueeze(0)
                # debug
                if args.debug:
                    print("Pred: ", pred_action)

                # update prev action
                prev_action = str(pred_action)

                if pred_action not in cls.TERMINAL_TOKENS:
                    # use predicted action and mask (if provided) to interact with the env
                    t_success, _, _, err, _ = env.va_interact(pred_action, interact_mask=pred_mask,
                                                              smooth_nav=args.smooth_nav,
                                                              debug=args.debug)
                    if not t_success:
                        fails += 1
                        if fails >= args.max_fails:
                            print("Interact API failed %d times" % (fails) + "; latest error '%s'" % err)
                            break

                # next time-step
                t_reward, t_done = env.get_transition_reward()
                reward += t_reward

                # update subgoals
                curr_subgoal_idx = env.get_subgoal_idx()
                if curr_subgoal_idx == eval_idx:
                    subgoal_success = True
                    break

                # terminal tokens predicted
                if pred_action in cls.TERMINAL_TOKENS:
                    print("predicted %s" % pred_action)
                    break

            # increment time index
            t += 1

        # metrics
        pl = float(t - len(expert_init_actions)) + 1  # +1 for last action
        expert_pl = len([ll for ll in traj_data['plan']['low_actions'] if ll['high_idx'] == eval_idx])

        s_spl = (1 if subgoal_success else 0) * min(1., expert_pl / (pl + sys.float_info.epsilon))
        plw_s_spl = s_spl * expert_pl

        # log success/fails
        lock.acquire()

        # results
        for sg in cls.ALL_SUBGOALS:
            results[sg] = {
                'sr': 0.,
                'successes': 0.,
                'evals': 0.,
                'sr_plw': 0.
            }

        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'subgoal_idx': int(eval_idx),
                     'subgoal_type': subgoal_action,
                     'subgoal_instr': subgoal_instr,
                     'subgoal_success_spl': float(s_spl),
                     'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
                     'subgoal_path_len_weight': float(expert_pl),
                     'reward': float(reward)}
        if subgoal_success:
            sg_successes = successes[subgoal_action]
            sg_successes.append(log_entry)
            successes[subgoal_action] = sg_successes
        else:
            sg_failures = failures[subgoal_action]
            sg_failures.append(log_entry)
            failures[subgoal_action] = sg_failures

        # save results
        print("-------------")
        subgoals_to_evaluate = list(successes.keys())
        subgoals_to_evaluate.sort()
        for sg in subgoals_to_evaluate:
            num_successes, num_failures = len(successes[sg]), len(failures[sg])
            num_evals = len(successes[sg]) + len(failures[sg])
            if num_evals > 0:
                sr = float(num_successes) / num_evals
                total_path_len_weight = sum([entry['subgoal_path_len_weight'] for entry in successes[sg]]) + \
                                        sum([entry['subgoal_path_len_weight'] for entry in failures[sg]])
                sr_plw = float(sum([entry['subgoal_path_len_weighted_success_spl'] for entry in successes[sg]]) +
                               sum([entry['subgoal_path_len_weighted_success_spl'] for entry in
                                    failures[sg]])) / total_path_len_weight

                results[sg] = {
                    'sr': sr,
                    'successes': num_successes,
                    'evals': num_evals,
                    'sr_plw': sr_plw
                }

                print("%s ==========" % sg)
                print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
                print("PLW SR: %.3f" % (sr_plw))
        print("------------")

        lock.release()

    def create_stats(self):
        '''
        storage for success, failure, and results info
        '''
        self.successes, self.failures = self.manager.dict(), self.manager.dict()
        self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': dict(self.successes),
                   'failures': dict(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'subgoal_results_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

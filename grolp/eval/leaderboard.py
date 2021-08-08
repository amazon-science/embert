import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

from eval_task import EvalTask
from grolp import AlfredPredictor
from grolp.envs.thor_env import ThorEnv
from scripts.generate_maskrcnn import MaskRCNNDetector, CustomImageLoader


class Leaderboard(EvalTask):
    '''
    dump action-sequences for leaderboard eval
    '''

    @classmethod
    def run(cls, predictor, image_loader, region_detector, task_queue, args, lock, splits, seen_actseqs,
            unseen_actseqs):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        try:
            while True:
                if task_queue.qsize() == 0:
                    break

                task = task_queue.get()

                try:
                    traj = predictor.load_task_json(task["split"], task)
                    r_idx = task['repeat_idx']
                    print("Evaluating: %s" % (traj['root']))
                    print("No. of trajectories left: %d" % (task_queue.qsize()))
                    cls.evaluate(env, predictor, r_idx, image_loader, region_detector, traj, args, lock, splits,
                                 seen_actseqs, unseen_actseqs)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print("Error: " + repr(e))
        finally:
            # stop THOR
            env.stop()

    @classmethod
    @torch.no_grad()
    def evaluate(cls, env: ThorEnv, predictor: AlfredPredictor, r_idx: int, image_loader: CustomImageLoader,
                 region_detector: MaskRCNNDetector, traj_data, args, lock, splits, seen_actseqs, unseen_actseqs):

        instruction_idx = 0 if predictor.is_split_model else None
        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)
        # goal instr
        high_descs = traj_data['turk_annotations']['anns'][r_idx]["high_descs"]
        done, success = False, False
        fails = 0
        t = 0
        actions = []
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        # used to determine whether we successfully execute an action
        is_action_ok = True
        pred_action = None
        pred_mask = None
        output_dict = None
        action_mask = None
        object_mask = None
        action_mems = None
        state_mems = None
        language_features = None
        language_masks = None
        prev_actions = None
        prev_objects = None

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

            # check if <<stop>> was predicted
            if pred_action == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            # print action
            if args.debug:
                print(pred_action)

            # use predicted action and mask (if available) to interact with the env
            is_action_ok, _, _, err, api_action = env.va_interact(
                pred_action, interact_mask=pred_mask, smooth_nav=False
            )

            if not is_action_ok:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # save action
            if api_action is not None:
                actions.append(api_action)

            # next time-step
            t += 1

        # actseq
        seen_ids = [t['task'] for t in splits['tests_seen']]
        actseq = {traj_data['task_id']: actions}

        # log action sequences
        lock.acquire()

        if traj_data['task_id'] in seen_ids:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        lock.release()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']

        files = seen_files if self.args.eval_split == "tests_seen" else unseen_files

        if self.args.debug:
            files = files[:2]

        for traj in files:
            traj["split"] = self.args.eval_split
            task_queue.put(traj)

        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        if args.debug:

            lock = self.manager.Lock()
            self.run(
                self.predictor, self.image_loader, self.region_detector, task_queue, self.args, lock,
                self.splits, self.seen_actseqs, self.unseen_actseqs
            )
        else:
            # start threads
            threads = []
            lock = self.manager.Lock()
            # self.model.test_mode = True
            for n in range(self.args.num_workers):
                thread = mp.Process(target=self.run, args=(
                    self.predictor, self.image_loader, self.region_detector, task_queue, self.args, lock,
                    self.splits, self.seen_actseqs, self.unseen_actseqs
                ))
                thread.start()
                threads.append(thread)

            for t in threads:
                t.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        storage for seen and unseen actseqs
        '''
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(self.seen_actseqs),
                   'tests_unseen': list(self.unseen_actseqs)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path,
                                 f'tests_{self.args.eval_split}_actseqs_dump_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="storage/data/alfred/splits/oct21.json")
    parser.add_argument('--data', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--eval_split', type=str, default='tests_seen', choices=['tests_seen', 'tests_unseen'])
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--cuda_device', dest='cuda_device', type=int, default=-1)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--rotation_steps', type=int, default=4, help="Assumes we're rotating 90 each time")
    ## MaskRCNN parameters
    parser.add_argument('--box_score_thresh', type=float, default=0.05)
    parser.add_argument('--box_nms_thresh', type=float, default=0.5)
    parser.add_argument('--panoramic_boxes', type=int, nargs="+", default=(36, 18, 18, 18))
    parser.add_argument('--max_boxes_per_image', type=int, default=36)
    parser.add_argument('--frame_size', type=int, default=300)
    parser.add_argument('--maskrcnn_checkpoint', default="storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt",
                        type=str)

    # parse arguments
    args = parser.parse_args()

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # leaderboard dump
    eval = Leaderboard(args, manager)

    # start threads
    eval.spawn_threads()

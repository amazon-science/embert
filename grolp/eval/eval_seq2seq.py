import argparse

import torch.multiprocessing as mp

from grolp.eval.eval_subgoals import EvalSubgoals
from grolp.eval.eval_task import EvalTask

if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="storage/data/alfred/splits/oct21.json")
    parser.add_argument('--data', type=str, default="storage/data/alfred/json_feat_2.1.0")
    parser.add_argument('--reward_config', default='configs/rewards.json')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cuda_device', dest='cuda_device', type=int, default=-1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--rotation_steps', type=int, default=4, help="Assumes we're rotating 90 each time")
    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10,
                        help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--horizon0', dest='horizon0', action='store_true',
                        help='Horizon0 robot perspective')
    parser.add_argument('--subgoals', type=str,
                        help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true',
                        help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true',
                        help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true',
                        help='no teacher forcing with expert')

    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--preprocess_workers', type=int, default=1)

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')
    parser.add_argument('--save_video_path', type=str)

    ## MaskRCNN parameters
    parser.add_argument('--box_score_thresh', type=float, default=0.05)
    parser.add_argument('--box_nms_thresh', type=float, default=0.5)
    parser.add_argument('--panoramic_boxes', nargs="+", default=(36, 18, 18, 18), type=int)
    parser.add_argument('--max_boxes_per_image', type=int, default=36)
    parser.add_argument('--frame_size', type=int, default=300)
    parser.add_argument('--maskrcnn_checkpoint', default="storage/models/vision/moca_maskrcnn/weight_maskrcnn.pt",
                        type=str)

    # parse arguments
    args = parser.parse_args()

    # eval mode
    if args.subgoals is not None:
        eval = EvalSubgoals(args, manager)
    else:
        eval = EvalTask(args, manager)

    # start threads
    eval.spawn_threads()

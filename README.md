# nexarm-imitation

Imitation learning on a NexArm leader–follower arm: 102 teleoperated
demonstrations of one task ("pick up the medicine box and put it in the bin"),
ACT trained on them, π0.5 (a pretrained vision-language-action model)
LoRA-fine-tuned on the same data on a single 16 GB GPU, and both evaluated on
the real arm under the same conditions.

![ACT and π0.5 on the same placement, side by side](docs/img/demo.gif)

Fixed placement 19 of 20: ACT on the left, π0.5 with real-time chunking (RTC)
on the right, 2× speed, clocks in real time. Only π0.5 succeeds here. This is a
favourable example. Five placements had only π0.5 succeed: ACT's recording of
placement 15 has the operator's hand in it, and on the other three π0.5 needed
25–33 s. Over all 20 placements: ACT 12/20, π0.5 17/20. A 53 s video
([scripts/make_demo.py](scripts/make_demo.py)) adds a placement both policies
solve, random placements, two failures and the results.

**Status:** complete as a second project: data collection, two policies, three
real-robot evaluation sessions, a random-placement test and a demo video. It
follows [nexarm-teleop-latency](https://github.com/lfulai223-wq/nexarm-teleop-latency),
which measured and reduced the teleoperation latency of the same arm. See
[what is not done](#limitations). Full write-up: [docs/report.md](docs/report.md).

## Results

20 fixed placements, one 40 s trial each. The three policies ran back to back in
one session (2026-11-18), with the same camera position:

| Policy | Box in the bin within 40 s | Within 25 s | Median time to release (successes) | Control rate |
|---|---|---|---|---|
| ACT, 102 demos | 12/20 | 10/20 | 12.2 s | 14.9 Hz |
| π0.5, synchronous inference | 17/20 | 15/20 | 15.3 s | 13.4 Hz |
| π0.5, RTC | 17/20 | 13/20 | 9.9 s | 15.0 Hz |

Success within 40 s is the operator's call. The 25 s column and the release
times come from the joint data: the moment the gripper opens with the base
over the bin. That rule agreed with the operator on all 120 trials of the six
fixed-placement sessions in this project.

- **π0.5 succeeds more often, but 20 placements cannot settle it.** Placement
  by placement, synchronous π0.5 succeeded where ACT failed on 7 placements and
  the reverse happened on 2 (exact sign test, p = 0.18). With RTC it is 5 to 0
  (p = 0.06).
- **The gap is in one orientation.** With the box lying horizontally in the
  camera image, ACT managed 2/7, π0.5 7/7 (synchronous) and 5/7 (RTC).
  Vertical and diagonal placements came out about the same for all three.
- **π0.5 is slow to predict.** One 50-step action chunk takes 245 ms on the
  RTX 5060 Ti (batch 1, with pre- and post-processing, LoRA merged into the
  weights); ACT takes 9 ms. With synchronous inference the arm stalls for about
  0.3 s every 2 s. RTC predicts the next chunk in a background thread: the
  stalls disappear, the success count stays the same, and successful trials
  finish sooner. Inference is no faster; RTC hides the 245 ms inside execution.

![Box-in-bin count against time for ACT with 45 and 102 demos, and for ACT and π0.5](docs/img/success_vs_time.png)

Each step is one placement whose box landed in the bin, at the moment the
gripper opened over it. **Left:** ACT trained on 45 demos against ACT trained
on 102, interleaved placement by placement, the operator not told which was
running. The first evaluation had found every failure at the grasp and 0/7 on
horizontal placements, so the second batch of demos added horizontal ones
(about 4 → 26). All the extra successes are horizontal placements. Paired: 5
placements to 1 within 40 s (p = 0.22), 5 to 0 within 25 s (p = 0.06).
**Right:** the session in the table.

![Frame-to-frame interval over one trial for each inference mode](docs/img/control_timing.png)

Interval between recorded frames, one per control step. Synchronous π0.5
stops for about 0.3 s each time it re-plans. Under RTC no interval in the whole
session exceeded 131 ms. The first interval in each trace is warm-up.

### Random placements

18 more trials of π0.5 with RTC, with the box put anywhere in the start area at
any angle instead of on the fixed placements:

| Setup | Trials | In the bin | Failures |
|---|---|---|---|
| Box flat, random position and angle | 15 | 12 | Missed grasp (box on a marker sheet at the edge of the start area); box knocked against the bin's front wall, fell outside; released on the bin's rim |
| Distractor objects next to it (a smaller box, a printed card) | 2 | 0 | Gripper never closed |
| Box standing on its end | 1 | 0 | Never lifted |

![Grasp positions on random placements against the training demos](docs/img/placement_coverage.png)

Where π0.5 grasped, against the 100 training demos that contain a grasp. The
joint angles at the first gripper close give the box's position (base angle)
and orientation (gripper angle on the table; 0° and 180° are the same). Camera
images could not be used: at the start pose the arm hides the box from the
stand camera. All 16 random trials with a grasp fall inside the training
coverage by a nearest-neighbour test on base, elbow and gripper angle; the
nearest demo is within 7° of base angle and 20° of gripper angle. These
placements test generalisation within the training coverage, not out of
distribution. The scenes that were really new, distractors and a standing box,
failed all three times.

## How it works

```
leader arm ── teleoperation, 15 Hz ──▶ follower arm      stand camera 640×360 ─┐
                                          │              wrist camera 640×480 ─┤
                                          └── 6 joint positions ───────────────┴─▶ LeRobot dataset
                                                                                    102 demos, 26,139 frames
                         ┌──────────────────────────────────────────────────────────────┴───────┐
                         ▼                                                                      ▼
   ACT, 52 M parameters: ImageNet ResNet18 image encoder,          π0.5 (lerobot/pi05_base), 4.1 B parameters:
   transformer predicts 30 actions (2 s) at a time                 LoRA, 28.7 M trainable (0.69%); flow matching
                                                                   denoises 50 actions in 10 steps
                         │                                                                      │
                         └──────────────▶ real arm, 20 fixed placements, 40 s each ◀────────────┘
```

- **Setup.** NexArm leader and follower (5 joints plus a gripper, bus servos),
  driven through the vendor's LeRobot fork. Two USB cameras on fixed exposure,
  curtains drawn. Recording and control run at 15 Hz: at 20 Hz the stand camera
  repeated 18% of frames, at 15 Hz none. ArUco markers on the desk are only
  there to tell whether a camera has been knocked; a check script compares
  their pixel positions with a reference before every session.
- **Data.** Two batches, 109 recorded, 102 kept (29.0 minutes). The 7 deleted
  had a skipped reset (operator in frame or arm idle), ended before the release,
  or started with the box already in the gripper. The second batch was planned
  from the first evaluation's failures, with a voice prompter announcing each
  placement because reading a table at the robot had led to misplacements.
- **ACT.** LeRobot's implementation with an action chunk of 30 steps. ACT's
  default of 100 is about 2 s at 50 Hz but would be 6.7 s of open-loop motion
  at 15 Hz. 100,000 steps at batch 8 in bf16, 3 h 27 min. Deployed as trained,
  re-planning after each full chunk; temporal ensembling and re-planning every
  10 steps looked worse in brief trials (two each) and were dropped.
- **π0.5.** Peak memory was measured for five fine-tuning options first. Full
  fine-tuning, and fine-tuning with only the vision encoder frozen, run out of
  memory at batch 1. LoRA fits: rank 16 on every linear layer of the language
  model and the action expert, the action and time projections fully trained,
  the vision encoder frozen. Learning rate 2.5e-4 with cosine decay to 2.5e-5,
  12,000 steps at batch 8, 14 h 47 min, 13.3 GB peak. For deployment the LoRA
  is merged into the weights (373 → 245 ms per chunk), and building the model
  skips random initialisation (load time 80 → 12 s, identical weights).
- **RTC** is LeRobot's implementation. The work here was the settings (next
  chunk requested when 20 actions remain, so it still re-plans about every
  2 s) and a serial bug RTC exposed. The background inference holds Python's
  GIL, so the vendor driver's 50 ms serial write timeout fired on writes that
  had succeeded. Raising it to 0.5 s fixed it: no errors in the 22 runs since.
- **Evaluation.** [scripts/eval_trials.py](scripts/eval_trials.py) runs each
  trial as a separate recorded rollout. Voice prompts announce the placement,
  keys score the trial, and errors are logged and the trial re-run. Two models
  can be interleaved per placement, alternating which goes first.
  [scripts/eval_stages.py](scripts/eval_stages.py) derives grasp and release
  times from the joint data and runs the paired sign test.
- **Safety.** Right after connecting, the follower's first servo reading is
  often garbage (wrist and gripper at the ends of their range). LeRobot stores
  that reading as the pose to return to after each trial, which would have
  twisted the wrist into its end stop at the end of about half the trials.
  [scripts/rollout_safe.py](scripts/rollout_safe.py) and
  [scripts/record_safe.py](scripts/record_safe.py) wrap the LeRobot commands
  and only start after two consistent good readings.

## Repository layout

| Path | Contents |
|---|---|
| [configs/](configs/) | LeRobot recording, practice teleoperation and rollout configs |
| [scripts/check_setup.py](scripts/check_setup.py) | Pre-session check: devices, camera rate and brightness, ArUco offsets, follower readings, leader–follower pose |
| [scripts/align_camera.py](scripts/align_camera.py) | Voice feedback for putting a knocked camera back |
| [scripts/record_safe.py](scripts/record_safe.py), [scripts/record_prompter.py](scripts/record_prompter.py) | Recording with the servo-reading guard; voice prompt for each placement |
| [scripts/pi05_train_memory.py](scripts/pi05_train_memory.py), [scripts/pi05_infer_latency.py](scripts/pi05_infer_latency.py) | π0.5 memory and time per fine-tuning option; prediction time |
| [scripts/train_pi05.py](scripts/train_pi05.py) | `lerobot-train` plus one processor-name alias needed to start from `lerobot/pi05_base` |
| [scripts/rollout_safe.py](scripts/rollout_safe.py) | `lerobot-rollout` with the servo-reading guard, LoRA merge, faster loading and per-frame timestamps |
| [scripts/eval_trials.py](scripts/eval_trials.py), [scripts/eval_stages.py](scripts/eval_stages.py) | Real-robot evaluation; joint-data analysis and paired comparison |
| [scripts/placement_coverage.py](scripts/placement_coverage.py) | Grasp position and orientation per trial, against the training demos |
| [scripts/export_results.py](scripts/export_results.py), [scripts/make_figures.py](scripts/make_figures.py), [scripts/make_demo.py](scripts/make_demo.py) | Result tables, figures, demo video and GIF |
| [results/](results/) | Every scored trial, the frame intervals and the grasp positions behind the numbers above |
| [docs/report.md](docs/report.md) | Full write-up |
| [docs/roadmap.md](docs/roadmap.md) | Plan and day-by-day log, including dead ends (Chinese) |
| [docs/recording_guide.md](docs/recording_guide.md) | Recording procedure (Chinese) |

The dataset, checkpoints and trial videos are not published because the camera
images show the room; [results/](results/) has the numbers.

## Running it

Everything except the figures runs in the environment of the vendor's LeRobot
fork (0.5.2, with `nexarm_follower` and `nexarm_leader`), with project 1
installed for the serial driver used by the setup check
(`pip install -e ~/nexarm-teleop-latency`). The figures need only numpy and
matplotlib.

```bash
V=<vendor LeRobot venv>/bin/python
$V scripts/check_setup.py                       # before every session
$V scripts/record_safe.py --config_path=configs/record_formal.yaml \
    --resume=true --dataset.num_episodes=56 --dataset.episode_time_s=25

ACCELERATE_MIXED_PRECISION=bf16 $V -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=local/nexarm_medbox_bin_clean_v2 --dataset.root=data/formal_clean_v2 \
    --policy.type=act --policy.chunk_size=30 --policy.n_action_steps=30 --policy.push_to_hub=false \
    --batch_size=8 --steps=100000 --save_freq=25000 --seed=1000 --output_dir=outputs/train/act_medbox_102ep
$V scripts/train_pi05.py ...                    # full command in docs/roadmap.md, 2026-11-17

$V scripts/eval_trials.py --duration 40 --rtc \
    --name pi05_12k_rtc --policy outputs/train/pi05_lora_102ep/checkpoints/012000/pretrained_model
$V scripts/eval_stages.py act102_100k_1118 pi05_12k_rtc --cut 25
$V scripts/export_results.py
python scripts/make_figures.py
```

`ACCELERATE_MIXED_PRECISION` is set because in this LeRobot version
`--policy.use_amp` only affects inference. The keys and voice prompts for
recording are in [docs/recording_guide.md](docs/recording_guide.md).

**Safety.** The follower moves under a learned policy. Keep hands out of its
workspace during a trial. `Esc` stops a trial at once and leaves the arm
holding its position.

## Limitations

- **Small samples.** 20 placements per policy. The direction is the same in
  every session, but no comparison reaches p < 0.05.
- **π0.5 against ACT was neither interleaved nor blind.** The three policies
  ran in consecutive blocks of one session, and synchronous π0.5 is
  recognisable by its stalls. Only the 45- against 102-demo ACT comparison
  was interleaved and blind.
- **One task, one object, one fixed scene.** The random placements stayed
  inside the training coverage. Distractors and a standing box were tried
  three times in all, and failed each time.
- **The camera moved between days** (knocked or re-plugged; 0.2° of rotation
  on Nov 17, 1.35° on Nov 18), so comparisons are made within one session.
- **Two operator interventions.** In ACT's trial on fixed placement 15 the
  operator laid a fallen box flat while the arm was running; ACT still failed,
  so this can only have helped ACT. In random trial 17 the operator took the
  box back out of the bin after the success. Neither changes a score.
- **The joint-data rule has a blind spot.** It sees where the gripper opens,
  not whether the box is still in it or where it falls. It would have passed
  two random-placement failures: one box had already fallen out of the
  gripper, the other was released on the rim and dropped outside. The
  operator's judgement is the score throughout.
- **Latency figures are per-prediction timings and frame intervals.**
  Camera-to-motion delay was not measured; with a static box it does not
  decide the outcome.
- **The π0.5 recipe differs from OpenPI's LoRA recipe**, which also trains the
  vision encoder; that does not fit in 16 GB here. The vendor's RTC guidance
  does not backpropagate through the network, unlike the RTC paper, and skips
  one action at each chunk switch.
- ACT and π0.5 predict chunks of different lengths (30 and 50); both re-plan
  about every 30 steps.

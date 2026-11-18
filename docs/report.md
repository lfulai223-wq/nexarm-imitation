# ACT and π0.5 on a desktop arm: report

The question: on one real task with one set of demonstrations, what does moving
from a small policy trained only on those demonstrations (ACT) to a large
pretrained vision-language-action model (π0.5) buy, and what does it cost?

**Short answer.** On 20 fixed placements, π0.5 put the box in the bin 17 times
and ACT 12 times, in the same session with the same 102 demonstrations. Nearly
all of the gap is on one orientation of the box. π0.5 takes 245 ms to predict
an action chunk against ACT's 9 ms. Run synchronously that stalls the arm every
2 s; with real-time chunking (RTC) the stalls go away, the success count stays
at 17/20, and successful trials finish sooner than ACT's. Twenty placements are
not enough to make the success difference statistically significant
(p = 0.18 synchronous, p = 0.06 RTC). On 15 random flat placements π0.5
succeeded 12 times, all within the range the demonstrations cover. It failed
all three trials with distractor objects or a standing box.

The day-by-day log, in Chinese and including dead ends, is
[roadmap.md](roadmap.md). Every scored trial is in
[../results/trials.csv](../results/trials.csv).

## 1. Task and setup

- **Task.** Pick up a medicine box (an ointment carton, about 14–15 cm long)
  from a start area on the desk and drop it into a bin. The start area is about
  15 × 15 cm, 15–30 cm from the arm's base. The box always lies printed side up
  and is gripped across its middle. The task string given to π0.5 is
  `Pick up the medicine box and put it in the bin.` It uses different words for
  the two boxes so the language model does not confuse them.
- **Arm.** NexArm leader and follower: 5 joints plus a gripper, bus servos,
  4096 counts per revolution. Driven by the vendor's fork of LeRobot (0.5.2).
  The stock SO-101 driver stops at the servo model check (shown in project 1).
- **Cameras.** A stand camera at 640×360 MJPG (640×480 crops about 12% off
  each side) and a wrist camera at 640×480 YUYV, its only mode. On auto
  exposure the wrist camera ran at 11 frames/s and washed out the white box
  (14.5% of white pixels clipped). Fixed exposure (25 ms plus a brightness
  offset) gives 22.8 frames/s with nothing clipped, so both cameras run on
  fixed exposure with the curtains drawn and the ceiling light on.
- **Rate.** Recording, teleoperation during recording, and policy control all
  run at 15 Hz. Sampling the latest frame at 20 Hz, the stand camera gave a new
  frame only 82% of the time; at 15 Hz, every time.
- **Fixed scene.** From the first real demonstration to the last evaluation,
  cameras, arm base, bin, markers, lighting, camera settings and the box itself
  stay unchanged. Four ArUco markers (4×4, 50 mm) around the start area are
  only used to detect a moved camera: [scripts/check_setup.py](../scripts/check_setup.py)
  compares their pixel positions with a stored reference before each session.
  It also checks devices, frame rates, image brightness against the reference,
  follower servo readings and the leader–follower pose difference.

## 2. Data

| | Recorded | Kept | Notes |
|---|---|---|---|
| Batch 1 (Nov 15) | 50 | 45 | 4 skipped resets (operator in frame, or the arm never moved), 1 ended before the release |
| Batch 2 (Nov 17) | 59 | 57 | 1 ended before the release, 1 started with the box already gripped |
| Total | 109 | 102 | 26,139 frames, 29.0 minutes |

- The first attempt at batch 1 was abandoned after 7 episodes. The wrist camera
  dropped off USB mid-episode, the recorder exited, and the vendor code turned
  the torque off, which drops an arm that is mid-motion. After reconnecting, a marker and the bin had
  moved slightly. All configs now keep the torque on when the program exits,
  and the wrist cable runs along the arm with slack at each joint.
- Batch 2 was planned from the first evaluation (section 3): two rounds of
  horizontal placements (the second rotated 5–15°), one round each of the
  other orientations, and extra episodes at the positions batch 1 had covered
  least. The final set has about 26 horizontal, 22 vertical, 21 diagonal and
  21 mirrored-diagonal demonstrations, plus 10 free placements.
- Reading a placement table at the robot led to two misplaced episodes, so
  [scripts/record_prompter.py](../scripts/record_prompter.py) follows the
  recorder's log and announces each placement by voice. The remaining 51
  episodes took 30 minutes with no misplacement.
- Every episode was checked from the joint data and its first frame, and
  questionable ones from the video.

## 3. ACT

**Configuration.** LeRobot's ACT: 52 M parameters, an ImageNet-pretrained
ResNet18 image encoder and a transformer that predicts a chunk of actions at
once. The chunk is 30 steps rather than the default 100. 100 steps is about
2 s at 50 Hz but would be 6.7 s of open-loop motion at 15 Hz; 30 steps is 2 s
again. Trained for 100,000 steps at batch 8, bf16, seed 1000: 3 h 27 min on an
RTX 5060 Ti 16 GB. Each 30-step chunk is executed in full before the next
prediction, as in training. One prediction takes 9 ms.

The training loss (L1 plus 10 × KL) said little: with 45 demos it bottomed at
0.110 at 35,000 steps and ended at 0.118. The real arm decided.

**First evaluation: 45 demos, 6/20.** 25 s per trial, 20 fixed placements
(section 5). All 14 failures stopped at the grasp: in none was the box carried
over the bin with the gripper closed. 3 never closed the gripper, and 11 closed
it 1–4 times without lifting the box. The successes grasped the box at 5–10 s
and released it over the bin 3–4 s later. So once ACT held the box, carrying
and releasing were not a problem. Horizontal placements went 0/7.
Batch 1 had about 4 horizontal demonstrations, against about 11 vertical and
20 diagonal.

**Inference variants (same model, two trials each, both horizontal).**
Temporal ensembling (predict every step, average the overlapping chunks) made
the arm slower and hesitant: median joint speed 105–128 counts/s against
180–375. One trial closed the gripper only at 18 s, the other never.
Re-planning every 10 steps never closed the gripper in 40 s. A possible reason
is that the close comes late in the chunk, so each new prediction starts the
approach again; this was not verified. Executing full chunks with 40 s per
trial closed the gripper 5 and 8 times without lifting. Two trials per variant
prove nothing on their own; they were stopped because the motion looked
clearly worse. With horizontal placements at 0/13 across all settings, the
answer was more data, not a different inference mode.

**45 against 102 demos (Nov 17).** The two models were interleaved: each
placement was run twice in a row, once per model, alternating which went
first. The voice prompts said only "round one" and "round two", not which
model was running.

| | 45 demos | 102 demos |
|---|---|---|
| In the bin within 25 s (joint data) | 6/20 | 11/20 |
| In the bin within 40 s (operator) | 8/20 | 12/20 |
| Median release time, successes | 17.7 s | 11.1 s |
| Horizontal / vertical / diagonal / mirrored | 0/7, 5/7, 2/5, 1/1 | 4/7, 5/7, 2/5, 1/1 |

Placement by placement: within 25 s, 5 placements succeeded only with 102 demos
and none only with 45 (exact sign test, p = 0.06); within 40 s, 5 against 1
(p = 0.22). The gain is entirely horizontal, where batch 2 added data. The
45-demo model scored 6/20 within 25 s again, as on the first day, so the setup
was stable across days. The 102-demo model went forward to the comparison with
π0.5.

## 4. π0.5

**What fits on 16 GB.** π0.5 (`lerobot/pi05_base`) has 4.1 billion parameters.
[scripts/pi05_train_memory.py](../scripts/pi05_train_memory.py) runs one
training step exactly as `lerobot-train` does (forward, backward, gradient
clipping, AdamW, gradient checkpointing) on synthetic inputs shaped like this
dataset:

| Fine-tuning | Trainable parameters | Peak memory at batch 8 | Largest batch | Time per sample |
|---|---|---|---|---|
| Everything | 4.14 B | — | out of memory at batch 1 | — |
| Vision encoder frozen | 3.73 B | — | out of memory at batch 1 | — |
| Action expert only | 0.69 B | 13.3 GB | 8 | 0.41 s |
| LoRA, LeRobot default targets | 1.3 M | 10.7 GB | 16 | 0.40 s |
| LoRA r=16 on all linear layers of both models | 26.6 M | 11.5 GB | 16 | 0.61 s |

Time per sample hardly depends on batch size (the GPU is saturated), so
training time depends only on the number of samples seen.

**Recipe.** LoRA rank 16 on the attention and MLP projections of both the
language model and the action expert. The action input/output projections and
the time MLP are fully trained, as in OpenPI's low-memory LoRA configuration.
The vision encoder is frozen; OpenPI trains it, but its gradients and
optimiser state (about 5 GB) do not fit. 28.7 M trainable parameters (0.69%).
Learning rate 2.5e-4 (ten times π0.5's default, as LeRobot's PEFT guide
suggests for LoRA) with cosine decay to 2.5e-5 after 400 warm-up steps; other
settings at π0.5 defaults. Action chunk 50. 12,000 steps at batch 8, about
3.7 passes over the data: 14 h 47 min, 4.4 s per step, 13.3 GB peak. The loss
(flow-matching MSE) fell from 2.04 to 0.065 and was still falling.

Two obstacles on the way:

- The tokenizer comes from `google/paligemma-3b-pt-224`, which is gated
  behind the Gemma licence. It needed a Hugging Face login after accepting
  the licence.
- The processor config of `lerobot/pi05_base` on the Hub names one step
  `relative_actions_processor`, while the vendor's LeRobot registers the same
  class as `delta_actions_processor`. [scripts/train_pi05.py](../scripts/train_pi05.py)
  registers the alias and otherwise is `lerobot-train`. The step is disabled
  here, so it does nothing either way.

`torch.compile` ran out of memory on the first step in both modes tried.

**Offline check before the real arm.** On 40 frames spread over the dataset,
predicting 30 steps as deployment would, mean error against the recorded
actions in counts per joint:

| | First step, median | Mean over 30 steps, median | Worst frame |
|---|---|---|---|
| ACT, 102 demos | 7, 4, 10, 10, 9, 1 | 11, 11, 15, 17, 17, 2 | 61 |
| π0.5 | 15, 14, 33, 37, 55, 11 | 20, 34, 55, 55, 68, 17 | 287 |

π0.5 fits the training frames less tightly. ACT saw each frame about 30 times
against π0.5's 3.7, and π0.5 samples its actions from noise. Still, the errors
are of the right magnitude (a normalisation bug shows up as hundreds or
thousands of counts), and on the real arm π0.5 did better.

**Deployment.** Prediction time at batch 1, with pre- and post-processing:

| | Step that predicts a new chunk | Other steps (from the queue) |
|---|---|---|
| ACT, 102 demos | 9 ms | 1.1 ms |
| π0.5, LoRA unmerged (LeRobot default) | 373 ms | 6.9 ms |
| π0.5, LoRA merged into the weights | 245 ms | 2.8 ms |

Merging takes 0.1 s, and the merged model's chunks differ from the unmerged
ones by at most 0.0074 in normalised units (about one count). The bare model
without LoRA or pre-processing predicts a 50-step chunk in 246 ms with 10
denoising steps, using 8.8 GB. Loading took 80 s, of which 73 s went into randomly initialising
4 billion parameters that the checkpoint then overwrites. Skipping that
initialisation brings loading to 12 s with identical weights (all 813 tensors
equal, identical chunks for the same noise). Both changes live in
[scripts/rollout_safe.py](../scripts/rollout_safe.py).

## 5. Evaluation protocol

- **Placements.** 20 fixed placements: five positions (four corners of the
  start area and the centre) × horizontal, vertical and diagonal (15), plus
  centre-left and centre-right × horizontal and vertical (4), plus the centre
  with the diagonal mirrored (1). Orientations are as seen in the stand
  camera. The same list is used for every model.
- **Trial.** [scripts/eval_trials.py](../scripts/eval_trials.py) runs each
  trial as its own `lerobot-rollout` with recording, in its own directory, so
  an abort loses one trial at most. A voice prompt announces the placement.
  The operator places the box and starts the trial with a key, then scores it
  after the follower returns to its start pose. Keys pressed during the trial
  are ignored. A trial that crashes is logged as an error and re-run, and a
  trial with a setup mistake can be marked for redo. 25 s per trial on the
  first day, 40 s afterwards.
- **Success.** The box ends up in the bin. The bin's walls hide its inside
  from the stand camera, so the operator judges, and the videos are kept for
  review.
- **Joint-data check.** The gripper counts as closed when its reading exceeds
  its opening value (the minimum over the first five frames) by 250 counts. A
  close-to-open transition with the base joint at 1509–1692 counts as a release
  over the bin; that range is the training demos' release range widened by 50.
  This gives a release time for every trial and a score at any cut-off, such as
  25 s within a 40 s trial. It agreed with the operator on all 120 trials of the
  six fixed-placement sessions. Times come from per-frame timestamps, because
  a control step that overruns records no frame: a 40 s π0.5 trial with
  synchronous inference has about 536 frames, and counting frames would make
  it look up to 4 s faster.
- **Comparisons.** Each placement pairs the two policies, and only placements
  where exactly one succeeded say anything, so the test is an exact sign test
  on those (McNemar's exact test).
  [scripts/eval_stages.py](../scripts/eval_stages.py) prints stages per trial
  and the paired test.

## 6. ACT against π0.5 (Nov 18)

π0.5 with synchronous inference (re-planning every 30 steps, ACT's rhythm)
ran first, then ACT, then π0.5 with RTC: three consecutive blocks of 20 in one
session at the same camera position, 10:19–11:54. This is not interleaved, and
it cannot be blind, because synchronous π0.5 visibly stalls.

| | ACT, 102 demos | π0.5, synchronous | π0.5, RTC |
|---|---|---|---|
| In the bin within 40 s (operator) | 12/20 | 17/20 | 17/20 |
| Within 25 s (joint data) | 10/20 | 15/20 | 13/20 |
| Horizontal / vertical / diagonal / mirrored | 2/7, 6/7, 3/5, 1/1 | 7/7, 6/7, 3/5, 1/1 | 5/7, 7/7, 4/5, 1/1 |
| Median release time, successes (range) | 12.2 s (8.5–36.9) | 15.3 s (4.5–29.4) | 9.9 s (7.3–33.1) |
| Control rate; frame gaps over 150 ms per trial | 14.9 Hz; about 1 | 13.4 Hz; about 18 | 15.0 Hz; none (max 131 ms) |

- **Paired.** Synchronous π0.5 against ACT: 7 placements to 2 within 40 s
  (p = 0.18), 8 to 3 within 25 s (p = 0.23). RTC against ACT: 5 to 0
  (p = 0.06), 3 to 0 within 25 s (p = 0.25). RTC against synchronous: 2 to 2
  (p = 1.00).
- **Where.** The difference is the horizontal placements, the orientation ACT
  had struggled with since the first evaluation. Synchronous π0.5's three
  failures were all at the grasp; with RTC, two were at the grasp and one
  held the box but never delivered it. One placement (top-right, diagonal)
  failed under all three.
- **ACT is stable across days.** ACT with 102 demos scored 12/20 on Nov 17 and
  again on Nov 18 (paired p = 1.00), although the camera had been re-aligned in
  between and the per-orientation split moved by a trial or two.
- **Synchronous π0.5 is slower to finish.** The arm stops for about 0.3 s each
  time a chunk is predicted, about 18 times per trial. Part of the slower
  completion is these stalls and part is slower motion.

## 7. Real-time chunking

LeRobot's RTC (`--inference.type=rtc`) predicts the next chunk in a background
thread while the control loop keeps executing every 67 ms. The first actions
of the new chunk are guided towards the not-yet-executed remainder of the
current one, like inpainting, so consecutive chunks join smoothly. It needs a
flow-matching policy, so it works for π0.5 but not ACT.

**Settings.** The next chunk is requested when 20 actions remain in the queue.
With 50-step chunks, a new image is used about every 30 steps (about 21
predictions per 40 s), the same rhythm as synchronous π0.5 and ACT. Execution
horizon 10, maximum guidance weight 10, exponential schedule: the values
LeRobot's RTC guide suggests.

**Offline check** at four points in the dataset, predicting the next chunk
after 20 executed steps with an assumed delay of 5 steps:

| | Gap between the new chunk's first 5 actions and the old chunk's remainder (counts, mean over the 5 actions, worst joint) | Time |
|---|---|---|
| No guidance (what synchronous inference does at every switch) | 95–344 | 254–257 ms |
| RTC guidance | 1–8 | 261–264 ms |

In this LeRobot version the guidance term is computed after the velocity, so
it does not backpropagate through the network as in the RTC paper; that is
why it costs only about 8 ms.

**Result.** 17/20, the same as synchronous inference, with no stalls
(13.4 → 15.0 Hz). Successes were quicker: median release 9.9 s, against 15.3 s
synchronous and 12.2 s for ACT. Four successes came after 25 s (two for
synchronous), so the 25 s count is two lower. The prediction itself still takes
245 ms; RTC moves it off the control loop.

**A serial bug RTC exposed.** 3 of the first 11 RTC runs exited with
`SerialTimeoutException: Write timeout` within the first second, before any
motion. Those trials were logged as errors and re-run. The vendor driver sets
a 50 ms write timeout, and pyserial checks it after writing a few bytes. While
the background thread runs inference it holds Python's GIL, so by the time the
main thread gets it back, 50 ms may have passed: the write itself had
succeeded. Synchronous inference runs in the same thread as the serial I/O,
which is why the 60 earlier synchronous runs never hit it. A model warm-up before
connecting helped but did not cure it; raising the write timeout to 0.5 s did.
Across the session, 4 of 24 runs errored, all at start-up, and none in the 22
runs since the change.

**Two quirks of the vendor's RTC, left as they are.** The inference delay is
computed as ceil(duration / 67 ms), which rounds 4 executed steps up to 5, so
one action is skipped at every chunk switch (the log warns each time). The
first chunk arrives while the queue is empty and the arm still, so its first
few actions are skipped too; the first command differs from the current pose
by up to 61 counts (45 for synchronous π0.5, 30 for ACT). No visible jolt.

## 8. Random placements

18 trials of π0.5 with RTC (18:26–19:10, 40 s each). The operator put the box
anywhere in the start area at any angle (`eval_trials.py --free`). 20 were
planned; the run stopped at 18 when Esc was pressed while trial 19 was
loading, before the arm was connected.

| Setup | Trials | In the bin | Failures |
|---|---|---|---|
| Box flat, random position and angle | 15 | 12 | Trial 3: box on a marker sheet at the edge of the start area, closed twice without lifting. Trial 5: gripped by one end, knocked against the bin's front wall on the way and fell outside; the gripper opened over the bin afterwards. Trial 12: released on the bin's near rim, fell outside |
| Distractor objects (a smaller coloured box; in trial 9 also a printed card) | 2 | 0 | The gripper never closed |
| Box standing on its end | 1 | 0 | Never lifted; the gripper closed once beside it |

Successes released after a median of 11.3 s (3.7–19.0). The loop ran at 15.0 Hz with a longest gap of 137 ms, and there
were no errors. Trial 16 (standing box) was scored a success at the time.
Reviewing the video and joint data showed the box never left the table, so it
was re-scored as a failure; the CSV records why.

**Were these placements new?** Camera images could not answer this: at the
start pose the arm hides most of the box from the stand camera, and the wrist
camera rarely sees it. Instead, [scripts/placement_coverage.py](../scripts/placement_coverage.py)
takes the joint readings at the first gripper close. The base angle gives the
box's direction and the elbow its distance. The gripper's angle on the table,
(wrist roll − base) mod 180°, gives its orientation: about 88° horizontal,
about 0° vertical. The same is computed for the 100 training demos that
contain a grasp. Each value is scaled (training standard deviation for base
and elbow, 52° for orientation), and a placement counts as new if its nearest
demo is farther than the 90th percentile of the demos' own leave-one-out
nearest distances. All 16 random trials with a grasp fall inside, the nearest
demo within 7° of base angle and 20° of orientation. The demonstrations
(7 positions × 4–5 orientations, plus free and slightly rotated placements)
already cover flat placements in the start area densely. These trials measure
generalisation within that coverage, not out of distribution. Distractors and
a standing box were genuinely new, and all three failed. For trials without a
good grasp the measure is where the policy tried to grasp, not where the box
was.

**Where the joint-data check disagreed.** It disagreed with the operator twice
in this session, after 120 agreements. In trials 5 and 12 the gripper opened at
base readings 1676 and 1674, inside the 1509–1692 range, with the box outside
the bin. All 84 successes across every evaluation opened between 1549 and 1655,
so the widened upper bound is too generous there. The range was left unchanged
and the blind spot recorded. In trial 17 the box was already under the gripper.
It closed in the first three frames, which pushed the old mean-of-five opening
value up and hid the close. The opening value is now the minimum of the first
five frames; re-running every earlier session left all outcomes unchanged.

## 9. Problems found along the way

| Problem | How it showed | What was done |
|---|---|---|
| Garbage first servo reading | Right after connecting, the follower often reads wrist and gripper at 4095 / 544 / 4095. LeRobot stores that reading as the pose to return to after a trial | [rollout_safe.py](../scripts/rollout_safe.py) and [record_safe.py](../scripts/record_safe.py) wait for two consistent good readings and re-read implausible ones. On the first evaluation day the guard caught a bad first reading in 11 of 21 runs |
| Follower servo read faults | Joints 4–6 read as 16385 / 544 / 8192 and reads take 25 ms instead of 5 ms: 40% of reads idle and 89% under load on one day | Reseating and power-cycling; the setup check polls the follower for a while before each session |
| Wrist camera dropping off USB | Once during each batch | Cable routed along the arm with slack; fixed exposure re-applied after every reconnect (the camera reverts to auto) |
| Stand camera knocked | 14 px and 1.7° off between trials 19 and 20 of the first evaluation | Trial 20 voided and re-run; [align_camera.py](../scripts/align_camera.py) speaks the pixel offset while the camera is adjusted (back to about 4.5 px, 0.2°) |
| Crash on an empty episode | A key pressed exactly at the reset timeout ended the next episode before its first frame; the vendor code crashes when saving it | record_safe.py ignores keys pressed before recording starts and skips empty episodes |
| Arm dropping on exit | The vendor default turns the torque off when the program exits, even mid-motion | `disable_torque_on_disconnect: false` in every config |
| Stalls missing from the recording | A control step that overruns records no frame, so frame counts understate elapsed time | Per-frame timestamps saved with every trial and used in all timing |
| Serial write timeout under RTC | See section 7 | Write timeout raised to 0.5 s |
| GPU shared during a trial | An unrelated offline experiment ran on the GPU during the last 7 s of RTC trial 8 | No visible effect (longest frame gap 81 ms); nothing else runs on the GPU during evaluations |

## 10. Limitations

- 20 placements per policy. The direction is consistent across sessions, but
  no comparison reaches p < 0.05.
- π0.5 against ACT ran in consecutive blocks, not interleaved, and could not
  be blind. Only the 45- against 102-demo comparison was interleaved and blind.
- One task, one object, one scene. Random flat placements stayed within the
  training coverage; distractors and a standing box were tried only three
  times.
- The camera position differed slightly between days (rotation 0.2° on Nov 17,
  1.35° on Nov 18). All comparisons are within one session.
- The operator intervened twice without changing a score. During ACT's trial
  on fixed placement 15 they laid a fallen box flat while the arm was running
  (ACT still failed). After the success in random trial 17 they took the box
  out of the bin while the policy was still running.
- The joint-data check cannot tell whether the box is still in the gripper,
  or where it lands.
- Latency is reported as per-prediction time and frame intervals. The
  camera-to-motion delay was not measured; with a static box it does not
  decide the outcome.
- The π0.5 recipe freezes the vision encoder, unlike OpenPI's LoRA recipe. The
  vendor's RTC differs from the paper (no backpropagation through the network)
  and skips one action per chunk switch.
- ACT and π0.5 use different chunk lengths (30 and 50); both re-plan about
  every 30 steps.

## 11. Next steps

- Re-run π0.5 against ACT interleaved, on more placements, with RTC (which
  removes the visible stall that prevented blinding).
- Add demonstrations with distractors and with the box standing, then test
  positions outside the start area, which would be genuinely new.
- Train the vision encoder as OpenPI does, on a larger GPU.
- For a task with moving objects, measure camera-to-motion delay with the
  high-speed video method from project 1.

# Concept notes: M15 learning from images

## 1. What changes when the input is pixels
The Stage 4 network received the cube's position as three numbers. Here it receives
2 × 64 × 64 × 3 = 24,576 numbers, almost all irrelevant, and has to find the few that matter.
The action space, the controller underneath, the expert and the evaluation are unchanged, so any
difference in results comes from perception being learned instead of given.

## 2. Teacher-student ("privileged") imitation: `learning/vision.py`, `learning/evaluate.py`
The teacher (ReactiveExpert) reads exact simulator state. The student sees only what a real robot
would: camera images and its own joint-derived gripper pose. During training the simulator pairs
each student observation with the teacher's action. This only works in simulation, where the
privileged state exists; it is one of the main reasons to train in simulation at all.

## 3. Two cameras, two jobs
- Front camera: the whole table. The cube is ~8 pixels: enough for "where roughly", not for grasping.
- Wrist camera: rides on the hand. From the home pose it sometimes cannot see the cube at all;
  just above the cube, the cube fills a large part of the image. Precision arrives when it is needed.

## 4. Convolutions and the spatial softmax
A convolution slides a small filter over the image; stacked layers build detectors for edges,
then coloured blobs, then objects. The last layer here produces 32 feature maps. A **spatial
softmax** turns each map into the (x, y) image coordinate of its strongest response: the network
ends up reporting 32 "keypoints" rather than a bag of features. For a task that is about *where*
things are, that is a useful built-in bias.

## 5. Why no random-crop augmentation
Shifting training images by a few pixels makes classifiers robust. Here the cameras are fixed and
the policy must convert pixels to millimetres; at the front camera one pixel is about 1 cm.
Random shifts would teach it a blurred geometry.

## 6. Precision is the bottleneck, and training budget buys precision
The grasp needs the gripper within ~3 mm of the cube. A policy whose prediction error is around
that size hovers just outside the expert's grasp trigger and freezes. Three consequences seen in
the experiments:
- more demonstrations → lower error → large jumps in success (Stage 4: 71% → 100%);
- adding DAgger data while keeping the number of epochs fixed made the fit *less* precise, and
  success fell; training longer on the same data brought it back;
- images make everything harder: the student's error starts ~10× higher than the state policy's.

## 7. Action chunking (ACT) in one paragraph
ACT predicts the next k actions at once and executes several before looking again. That gives
the policy commitment over a short horizon, which helps with pauses and multi-modal
demonstrations (the problems Stage 4 solved by redesigning the expert). It needs a much larger
network (18.7 M parameters here vs 0.2 M) and a GPU. `scripts/lerobot_act.py` runs it through
LeRobot; in this repository it has only been smoke-tested.

## Check your understanding
1. The student gets the gripper's position as numbers but not the cube's. Why is that a fair
   model of a real robot?
2. From the home pose the wrist camera sometimes sees no cube. What must the policy have learned
   for those episodes to succeed anyway?
3. Why did three fine-tuning epochs after DAgger make the student much worse, when the new data
   was correctly labelled?
4. Rendering without shadows and reflections made training feasible on CPU. What would you
   expect to happen if the trained policy were then run WITH shadows, and how would you test it?

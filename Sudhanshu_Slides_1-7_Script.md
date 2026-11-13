# IVP Seminar — Part 1 (Slides 1–7)

**Presenter:** Sudhanshu (IIT2024081)
**Topic:** Abnormal Behavior Detection in Multi-Camera Surveillance Videos (Group 19)
**Time:** about 6–7 minutes

> **Before you present:** on slide 1, the big title overlaps the "IVP Seminar #1 / Group No. 19 / Based on papers…" lines. Check it in PowerPoint and fix it if it looks the same on your screen.

---

## Part A: What your slides mean

Your job is to set up the topic. You explain what the problem is, how the field got here, and why multiple cameras matter. After that, each teammate goes deep into one paper.

### Slide 1: Title
The topic is automatically spotting unusual behaviour in CCTV footage from many cameras. The deck is based on five recent papers, and seven people present it.

### Slide 2: Outline
This is the roadmap. You and Akshat cover the background: what "abnormal" means, the history, challenges, datasets and metrics. Next, each of the five papers gets its own section. The seminar ends with a comparison, demos and the conclusion.

### Slide 3: The five papers in one line each
- **[1] Veesam et al., 2025:** a complete multi-camera system. It builds a graph of how people interact. Reinforcement learning (RL) decides which camera deserves the most processing power, and a fast "event" encoder catches sudden motion such as stampedes.
- **[2] Luo et al., AAAI 2023:** crowd-level anomalies such as counter-flow and crowd turbulence. In these, each person looks normal on their own, but the crowd as a whole moves wrongly. The paper measures how consistently the crowd moves, at several zoom levels.
- **[3] Cao et al., CVPR 2023:** introduces a new dataset, NWPU Campus (43 scenes, 28 anomaly types). It also adds two ideas. The first is *scene-dependent* anomalies: cycling is fine on a road but not on a footpath. The second is *anticipation*: predicting an anomaly before it happens.
- **[4] Pereira & Maia, 2023:** uses only video-level labels ("this video contains something abnormal") and combines the training loss across two overlapping cameras.
- **[5] Zanella et al., CVPR 2024:** needs no training at all. It writes a caption for each frame, and a large language model (LLM) rates how suspicious the scene sounds.
- **Why these five:** together they cover the main approaches in the field: multi-camera, crowd-level, scene-dependent, weakly supervised and training-free.

### Slide 4: What abnormal behaviour detection is
- The system learns what *normal* looks like. Anything rare, or anything that deviates from normal, gets flagged.
- The output is an **anomaly score for each frame**. A high score means abnormal, and a threshold decides when to raise an alert.
- Why automate it: humans watching CCTV get tired, judge inconsistently, and can't watch hundreds of feeds at once.
- Multiple cameras help because overlapping views remove blind spots.

### Slide 5: How the field evolved
- **2009–2012, hand-crafted:** people modelled the crowd with physics-like "social forces", or modelled motion texture statistically on the UCSD dataset.
- **2016–2018, deep normality:** a network learns to reconstruct or predict the next frame. If the prediction is bad, the frame is probably abnormal.
- **2018, weak supervision:** Sultani et al. released UCF-Crime and trained with only video-level labels, using multiple-instance learning (MIL).
- **2020–2023, structure and memory:** graphs, memory modules and self-supervised tricks.
- **2022–2023, new benchmarks:** UBnormal and NWPU Campus.
- **2024 onwards, foundation models:** large pretrained models such as captioners and LLMs used without any training.

### Slide 6: Why multiple cameras
One camera has three weaknesses: people hide each other, events happen outside or at the edge of the frame, and depth is hard to judge from one angle. Multiple cameras address each one:
- **Overlapping views:** paper [4] shows a second view raises F1 from about 43% to 75%.
- **Cross-view tracking:** the same person is followed across cameras.
- **Smart compute:** paper [1] spends processing effort only on the camera where something risky is happening.

### Slide 7: The six key challenges, and which paper tackles each
This slide connects your part to the rest of the seminar:

| Challenge | Tackled by |
|---|---|
| Occlusion and clutter | [1], [4] (multiple views) |
| Rare, open-ended anomalies | [3], [5] |
| Scene dependence | [3] |
| Varying scale (crowd vs. one person) | [2] |
| Costly frame-level labels | [4] (weak labels), [5] (no labels) |
| Real-time compute on many HD streams | [1] |

---

## Part B: Script

### Slide 1: Title (about 40 sec)
Good morning everyone. We are Group 19, and our seminar topic is **Abnormal Behavior Detection in Multi-Camera Surveillance Videos**.

This presentation is based on five recent research papers, published between 2023 and 2025 in venues like CVPR, AAAI and Scientific Reports.

There are seven of us, and each person will cover one part. I'm Sudhanshu, and I'll start with the background and the motivation behind this problem.

### Slide 2: Outline (about 40 sec)
Here is how the seminar is organised.

First, we'll build the background: what counts as abnormal behaviour, how this field has evolved, the main challenges, and the datasets and metrics researchers use.

Then we'll go through the five papers one by one, starting with Veesam et al. and ending with Zanella et al.

Finally, we'll compare all five papers, show some demos, discuss future directions, and conclude.

### Slide 3: Selected papers (about 1 min 15 sec)
These are the five papers we selected.

The first, by **Veesam et al.**, is a complete multi-camera framework. It models how people interact using graph attention, and it uses reinforcement learning to decide which camera needs the most attention.

The second, by **Luo et al.** at AAAI 2023, focuses on **crowd-level** anomalies like counter-flow and crowd turbulence, where individual people look normal but the crowd as a whole behaves dangerously.

The third, by **Cao et al.** at CVPR 2023, introduces a new dataset called **NWPU Campus**. It also brings in two new ideas: scene-dependent anomalies, and **anticipating** an anomaly before it actually happens.

The fourth, by **Pereira and Maia**, uses weak labels and combines information from **multiple overlapping camera views**.

The fifth, by **Zanella et al.** at CVPR 2024, is completely **training-free**. It uses image captions and a large language model to decide what is abnormal.

We chose these five because together they cover the main approaches in this field today.

### Slide 4: What is it (about 1 min 10 sec)
So what exactly is abnormal behaviour detection?

The goal is to automatically flag events in a video that are **rare**, or that **differ from the normal patterns** the system has learned.

In practice, the model outputs an **anomaly score for every frame**. A high score means the frame is likely abnormal, and if the score crosses a threshold, an alert is raised.

Typical examples are fighting, theft, sudden running, people moving against the crowd, stampedes, or a vehicle taking a wrong turn.

Why automate this? Because a human operator watching CCTV gets tired, judges subjectively, and simply cannot monitor hundreds of camera feeds at the same time.

And when we have **multiple cameras with overlapping views**, we can cover the blind spots and occlusions that a single camera misses.

The applications are everywhere: smart cities, railway stations and airports, large public events, and campuses like ours.

### Slide 5: Evolution (about 1 min 15 sec)
This timeline shows how video anomaly detection has evolved.

Around **2009 to 2012**, methods were **hand-crafted**. For example, the social force model treated a crowd like particles pushing on each other, and dynamic texture models captured motion patterns on the UCSD dataset.

From **2016 to 2018**, deep learning came in. Autoencoders and **future-frame prediction** learn what normal looks like: if the model cannot predict the next frame well, that frame is probably abnormal.

In **2018**, Sultani and colleagues introduced **weak supervision** using multiple instance learning on the UCF-Crime dataset. With this, you only need to know whether a whole video contains an anomaly, not exactly which frame.

From **2020 to 2023**, researchers added **graphs, memory modules and self-supervision** to capture structure and interactions.

Around the same time, **new benchmarks** like UBnormal and NWPU Campus appeared.

And from **2024**, we see **foundation models**: large language models doing anomaly detection without any training at all, as in our fifth paper.

### Slide 6: Why multiple cameras (about 1 min)
Now, why multiple cameras?

On the left are the problems with a **single camera**. In a dense crowd, one person can completely **hide** another. The field of view is **limited**, so events can happen at the edge or outside the frame. And from one angle, **depth and motion are ambiguous**.

On the right is what **multiple cameras** give us. With **overlapping views**, another camera can see what one camera misses; Pereira and Maia show this clearly. We can do **cross-view tracking**, re-identifying the same person across cameras. And we can use **smart compute**: as in Veesam et al., the system spends its processing power on the camera where something risky is happening, instead of treating all cameras equally.

### Slide 7: Key challenges (about 1 min)
Finally, the key challenges in this area. Each of our five papers tackles some of them.

**Occlusion and clutter**: people hide each other in crowds, which is why multi-view methods matter.

**Rare, unbounded anomalies**: abnormal events are scarce, and we can never list every type in advance.

**Scene dependence**: cycling is normal on a road but abnormal on a footpath. The same action can be normal or abnormal depending on where it happens.

**Varying scale**: something like crowd turbulence involves many people at once, not just one region.

**Costly labels**: marking every frame is slow and expensive, so most methods use weak labels.

And **real-time compute**: many HD camera streams must be processed with very low delay.

With this background, I'll hand over to **Akshat Parmar**, who will explain the types of abnormal behaviour and the learning paradigms used to detect them.

---

## Part C: Questions you might get, with short answers

- **"Anomaly vs. abnormal behaviour?"** An anomaly is anything that deviates from normal, which includes objects like a truck on a footpath. Abnormal *behaviour* is the subset caused by people's actions.
- **"Why not just train a normal classifier?"** Abnormal examples are rare and the types are open-ended. You can't collect every kind of anomaly, so most methods learn "normal" and flag deviations instead.
- **"What is MIL?"** Multiple-instance learning. Each video is a "bag" of clips with a single label. The model learns that the highest-scoring clip in an abnormal video should score above every clip in a normal video.
- **"How is performance measured?"** Mostly frame-level AUC: the area under the ROC curve, computed across all thresholds. Akshat covers the metrics on slide 12.
- **"Is the multi-camera data in paper [1] real?"** Mostly simulated: the authors split single-camera videos into overlapping views. Paper [4] uses genuinely overlapping cameras (PETS-2009). Saumya and Vishwas cover the details.

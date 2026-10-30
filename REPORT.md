# FreshSense AI — Full Report (Baby Steps)

> **Goal:** build an AI that looks at a photo of a fruit and tells you if it is
> **fresh** or **rotten**, using a simple Artificial Neural Network (ANN),
> reaching **at least 70% accuracy**, with a web demo.

**Final result: test accuracy = 79.4% (target ≥ 70%)** ✅

---

## Step 0 — What are we even building? (The big picture)

Imagine you have a basket of apples. Some are fresh, some have gone bad. You want
a phone app where you take a photo and it says "fresh ✅" or "rotten 🤢".

That is exactly FreshSense AI:
1. We collect **thousands of labelled photos** (each photo tagged "fresh" or "rotten").
2. We **train** a small neural network on them (the computer learns the difference).
3. We wrap it in a **website** (Streamlit) where anyone can upload a photo and get an answer.

Three ingredients: **dataset → model → demo**. Everything below explains each one
like you're five.

---

## Step 1 — The dataset (the model's textbook)

**What is a dataset?** A big pile of examples the model learns from — like a
textbook with questions AND answers.

- **Source:** `Project-AgML/fresh_rotten_fruit_classification` (free, from Hugging Face).
- **Contents:** photos of fruits (apples, bananas, oranges...), each labelled:
  - `0` = fresh 🍏
  - `1` = rotten 🤢
- **Size:** 2,400 images (1,600 fresh, 800 rotten; apples, bananas, grapes, guavas, jujubes, oranges, pomegranates, strawberries).
- **Why this dataset?** Fresh vs rotten is mostly a *colour/texture* difference
  (brown spots, dull skin). That makes it learnable for even a tiny network —
  which is why a *simple* ANN is enough here.

**Baby-step summary:** the dataset is the "answer key" the model studies.

---

## Step 2 — Preparing the photos (making them digestible)

A neural network cannot eat a JPEG. It only understands **numbers**. So:

1. **Open** each photo.
2. **Convert to RGB** (3 colour channels: red, green, blue).
3. **Resize to 48×48 pixels** — small enough to be fast, big enough to keep the
   brown spots visible.
4. **Divide every pixel by 255** — pixels are 0–255; dividing squashes them to
   0–1, which helps the network learn faster and more stably.
5. **Split** the pile: 80% for training, 20% kept hidden for the final test
   (stratified, so both piles have the same fresh/rotten ratio).

After this, one photo = a list of 48×48×3 = **6,912 numbers** between 0 and 1,
plus its label (0 or 1).

**Baby-step summary:** photos → small grids of numbers between 0 and 1.

---

## Step 3 — The model (a simple ANN, nothing fancy)

ANN = Artificial Neural Network: layers of tiny computing units ("neurons")
connected together. Ours is a plain **feedforward** network (data flows one way:
in → hidden layers → answer):

```
Input:  6,912 numbers (the 48×48×3 photo)
   ↓
Dense(256, relu)  + Dropout(0.3)     ← hidden layer 1: learns simple patterns
   ↓                                    (edges, colour blobs)
Dense(128, relu)  + Dropout(0.3)     ← hidden layer 2: combines them
   ↓                                    (spots, dull patches)
Dense(1, sigmoid)                    ← output: ONE number 0–1
                                       = probability the fruit is ROTTEN
```

- **Dense** = every neuron talks to every neuron in the next layer (the classic
  ANN building block from your coursework).
- **relu** = "if negative, output 0; else pass through" — lets the network learn
  non-trivial patterns.
- **sigmoid** = squashes the final score into 0–1, readable as a probability.
- **Dropout** = during training, randomly switch off 30% of neurons each step so
  the network can't memorise — it must *generalise*.

**Decision rule:** probability ≥ 0.5 → "rotten", else "fresh".

**Baby-step summary:** 6,912 numbers in → two hidden layers think → one
probability out.

---

## Step 4 — Training (how the model learns)

1. Show the network a **batch** of 128 photos, let it guess.
2. Compare guesses to true labels → compute the **loss** (binary cross-entropy:
   a number saying "how wrong you were").
3. **Backpropagation**: figure out which connections caused the error.
4. **Adam optimizer** nudges every connection a tiny step (learning rate 0.001)
   in the direction that reduces the error.
5. Repeat for the whole training set = **1 epoch**. We train up to 40 epochs.
6. After each epoch, check accuracy on a **validation slice** (15% of training
   data the model never trains on).
7. **Early stopping**: if validation accuracy stops improving for 8 epochs,
   stop and keep the best version (prevents overfitting = memorising).

**Baby-step summary:** guess → measure error → fix connections → repeat until
it stops getting better.

---

## Step 5 — Testing (the honest exam)

The **test set** (20% of photos) was locked away during training — the model has
*never* seen these images. We ask it to label them and count correct answers:

> **Test accuracy = 79.4%** (requirement: ≥ 70%) ✅

We also print a **confusion matrix** (how many fresh were called rotten and
vice versa) and a **classification report** (precision/recall per class).

**Baby-step summary:** exam on unseen photos → score = accuracy.

---

## Step 6 — The demo (Streamlit website)

`app.py` is a tiny web app (Streamlit):

1. User **uploads** a fruit photo.
2. App resizes it to 48×48, converts to numbers (same as Step 2).
3. The trained model (`freshsense_ann.keras`) outputs the rotten-probability.
4. Page shows **FRESH ✅** or **ROTTEN 🤢** + a freshness meter.

Run locally: `streamlit run app.py`
Deploy free: push to GitHub → https://share.streamlit.io → "New app" → pick the
repo → done, you get a public demo link.

**Baby-step summary:** upload photo → same number-crunching → verdict on screen.

---

## Step 7 — Project files

```
freshsense-ai/
├── train.py              # steps 1–5: load data, train ANN, evaluate, save model
├── app.py                # step 6: Streamlit demo
├── freshsense_ann.keras  # the trained model (created by train.py)
├── data/                 # dataset parquet files (downloaded once)
├── requirements.txt      # pip install -r requirements.txt
├── REPORT.md             # this file
└── README.md             # quick-start for GitHub
```

---

## Glossary (words used above, in one line each)

- **ANN** — network of neuron layers that learns patterns from examples.
- **Neuron** — takes numbers in, multiplies by weights, adds bias, applies relu.
- **Weight** — a connection's strength; *learning = tuning weights*.
- **Epoch** — one full pass over the training data.
- **Batch** — a small chunk (128) of photos used for one learning step.
- **Loss** — single number measuring "how wrong"; training minimises it.
- **Backpropagation** — the algorithm that assigns blame to each weight.
- **Overfitting** — memorising training photos; fixed by dropout + early stopping.
- **Accuracy** — % of test photos labelled correctly.

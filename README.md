# Poetic Text Generation ✍️

A character-level text generation model built with **TensorFlow/Keras**, trained on Shakespeare's writing to generate Shakespeare-style ("poetic") text one character at a time using an **LSTM (Long Short-Term Memory)** network.

## 📖 Overview

This project trains a neural network to predict the **next character** in a sequence, given the previous 40 characters. Once trained, the model can generate new, original text in a Shakespearean style by repeatedly predicting and appending one character at a time.

## 🧠 How It Works

1. **Data**: The model downloads Shakespeare's complete works (`shakespeare.txt`) from TensorFlow's public dataset storage, and uses a 10,000-character slice (`text[10000:20000]`) of it as training data.
2. **Preprocessing**: The text is lowercased, and every unique character in it is mapped to an integer index (`char_to_index`) and back (`index_to_char`).
3. **Sequencing**: The text is split into overlapping sequences of **40 characters** (`SEQ_LENGTH = 40`), with a step size of 3, where each sequence is paired with the character that comes right after it (the target/label).
4. **One-hot Encoding**: Each sequence and target character is converted into a one-hot encoded boolean array so it can be fed into the LSTM.
5. **Model Architecture**:
   - `LSTM(128)` layer that reads the character sequence
   - `Dense` layer with one unit per unique character
   - `Softmax` activation to output a probability distribution over the next character
6. **Training**: The model is compiled with `categorical_crossentropy` loss and the `RMSprop` optimizer (`learning_rate=0.01`), then trained for 4 epochs with a batch size of 256.
7. **Saving/Loading**: After training, the model is saved as `model.keras`. On future runs, if this file already exists, the script skips training and loads the saved model directly — saving time.
8. **Text Generation**: A `generate_text()` function starts from a random point in the source text and predicts characters one at a time, feeding each new predicted character back into the input for the next prediction (a sliding window).
9. **Sampling with Temperature**: The `sample()` function controls the randomness/creativity of predictions using a **temperature** parameter:
   - **Low temperature (e.g. 0.2)** → more predictable, conservative, repetitive text
   - **High temperature (e.g. 1.0)** → more random, creative, and sometimes less coherent text

## 📁 Project Structure

```
Poetic-Text-Generation/
├── main.py          # Main script: data prep, training, and text generation
├── model.keras       # Saved trained model (auto-generated after first run)
└── README.md          # Project documentation
```

## ⚙️ Requirements

- Python 3.x
- TensorFlow
- NumPy

Install dependencies with:

```bash
pip install tensorflow numpy
```

## 🚀 Usage

Simply run the script:

```bash
python main.py
```

- **First run**: Since `model.keras` doesn't exist yet, the script will download the dataset, train the LSTM model (4 epochs), and save it as `model.keras`.
- **Subsequent runs**: The script detects the existing `model.keras` file and loads it directly, skipping training.

After the model is ready, the script automatically generates and prints **5 samples of 300-character text**, each at a different temperature (`0.2`, `0.4`, `0.6`, `0.8`, `1.0`), so you can compare how creativity/randomness affects the output.

## 📝 Example Output (illustrative)

```
----------------
[generated text at temperature 0.2 — more coherent, repetitive]
----------------
[generated text at temperature 0.6 — balanced]
----------------
[generated text at temperature 1.0 — more random/creative]
----------------
```

## 🔧 Customization

- **Change training data size**: Modify the `text[10000:20000]` slice to use more or less of Shakespeare's text (more data = better results but longer training time).
- **Change sequence length**: Adjust `SEQ_LENGTH` to change how much context the model uses to predict the next character.
- **Train longer**: Increase `epochs` in `model.fit()` for potentially better results.
- **Generate different lengths**: Change the `length` argument passed to `generate_text()`.
- **Retrain from scratch**: Delete `model.keras` and re-run `main.py`.

## 💡 Key Concepts Used

- Character-level language modeling
- Recurrent Neural Networks (LSTM)
- One-hot encoding
- Temperature-based sampling for text generation
- Model persistence (save/load) with Keras

## 📌 Notes

- Training data is intentionally kept small (10,000 characters) for faster experimentation on limited hardware.
- The model is a simple single-layer LSTM — a great starting point for learning about sequence models and text generation, and a good base to build on (e.g. stacking more LSTM layers, adding dropout, or using GRUs).


import streamlit as st
import openai
import tempfile
import subprocess
import torch
import torchvision
import torchvision.transforms as transforms
import os
import time

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🤖 AI That Improves Itself")

def generate_model_code():
    prompt = """
    Create a PyTorch model that classifies MNIST digits. Include:
    - A model class named 'Net'
    - A function 'train_model()' that trains and returns test accuracy
    - Use torch, torchvision, and nn modules only
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert ML engineer. Return only executable PyTorch code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

def evaluate_model(code: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
            tmp.write(code.encode("utf-8"))
            tmp_path = tmp.name
        result = subprocess.run(["python", tmp_path], capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        if "ACCURACY:" in output:
            acc = float(output.split("ACCURACY:")[-1].strip())
            return acc, code
        return 0.0, "# Failed to extract accuracy."
    except Exception as e:
        return 0.0, f"# Error: {e}"

best_code = ""
best_acc = 0.0
N = 3
if st.button(f"Start Self-Improving AI (Top {N} Iterations)"):
    with st.spinner("Training and evaluating..."):
        for i in range(N):
            st.write(f"**Iteration {i+1}/{N}**")
            code = generate_model_code()
            acc, result = evaluate_model(code)
            st.code(code)
            st.write(f"Accuracy: {acc}")
            if acc > best_acc:
                best_acc = acc
                best_code = code
    with open("best_model.py", "w") as f:
        f.write(best_code)
    st.success(f"Best model saved with accuracy {best_acc}")

if os.path.exists("best_model.py"):
    st.subheader("🧠 Final Model Interface")
    import best_model
    from PIL import Image
    import numpy as np

    uploaded = st.file_uploader("Upload a digit image (28x28 grayscale)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("L").resize((28, 28))
        arr = transforms.ToTensor()(img).unsqueeze(0)
        pred = best_model.predict(arr)
        st.image(img, caption="Input Image", width=150)
        st.success(f"Predicted Digit: {pred}")

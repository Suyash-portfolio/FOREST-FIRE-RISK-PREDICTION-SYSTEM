const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = {
    temp: Number(document.getElementById("temp").value),
    RH: Number(document.getElementById("RH").value),
    wind: Number(document.getElementById("wind").value),
    rain: Number(document.getElementById("rain").value),
  };

  resultBox.textContent = "Predicting...";
  resultBox.className = "result";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    if (data.prediction === 1) {
      resultBox.textContent = "High Risk 🔥";
      resultBox.classList.add("high-risk");
    } else {
      resultBox.textContent = "Low Risk ✅";
      resultBox.classList.add("low-risk");
    }
  } catch (error) {
    resultBox.textContent = error.message;
  }
});

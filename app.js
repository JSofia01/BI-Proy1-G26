async function makePrediction() {
    const opinion = document.getElementById("opinionText").value;

    if (!opinion) {
        alert("Por favor ingrese una opinión.");
        return;
    }

    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify([{ 'TextosT': opinion }])
    });

    const result = await response.json();
    displayResult(result);
}

function displayResult(result) {
    const resultDiv = document.getElementById("result");
    const prediction = result.predictions[0];
    const probability = result.probabilities[0];

    resultDiv.innerHTML = `
        <h3>Resultado:</h3>
        <p><strong>ODS Predicho:</strong> ${prediction}</p>
        <p><strong>Probabilidad:</strong> ${probability}</p>
    `;
}

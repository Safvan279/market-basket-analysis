document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    displayResults(result);
});

function displayResults(result) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = ""; // Clear previous results

    if (result.error) {
        resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
        return;
    }

    // Display rules
    if (result.rules.length > 0) {
        const rulesTable = document.createElement('table');
        rulesTable.innerHTML = `
            <tr>
                <th>Rule</th>
                <th>Support</th>
                <th>Confidence</th>
                <th>Lift</th>
            </tr>
        `;
        result.rules.forEach(rule => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${Array.from(rule.antecedents).join(', ')} -> ${Array.from(rule.consequents).join(', ')}</td>
                <td>${rule.support.toFixed(4)}</td>
                <td>${rule.confidence.toFixed(4)}</td>
                <td>${rule.lift.toFixed(4)}</td>
            `;
            rulesTable.appendChild(row);
        });
        resultDiv.appendChild(rulesTable);
    } else {
        resultDiv.innerHTML = "<p>No rules found.</p>";
    }

    // Display model results
    const modelResultsDiv = document.createElement('div');
    modelResultsDiv.innerHTML = "<h2>Model Results</h2>";
    for (const [model, score] of Object.entries(result.model_results)) {
        modelResultsDiv.innerHTML += `<p>${model}: ${score}</p>`;
    }
    resultDiv.appendChild(modelResultsDiv);
}
    
document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    const resultsOutput = document.getElementById('results-output');
    const loader = document.getElementById('loader');
    const legendContainer = document.getElementById('legend');

    const entityColors = {
        'PER': { name: 'Person', color: 'var(--color-per)' },
        'ORG': { name: 'Organization', color: 'var(--color-org)' },
        'LOC': { name: 'Location', color: 'var(--color-loc)' },
        'MISC': { name: 'Miscellaneous', color: 'var(--color-misc)' }
    };

    // Populate the legend
    function populateLegend() {
        legendContainer.innerHTML = '';
        for (const key in entityColors) {
            const item = entityColors[key];
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.innerHTML = `
                <div class="legend-color" style="background-color: ${item.color};"></div>
                <span>${item.name}</span>
            `;
            legendContainer.appendChild(legendItem);
        }
    }
    
    // Function to show/hide loader
    const showLoader = (show) => {
        if (show) {
            loader.classList.add('show');
            analyzeBtn.disabled = true;
        } else {
            loader.classList.remove('show');
            analyzeBtn.disabled = false;
        }
    };

    // Function to render results
    const renderResults = (data) => {
        resultsOutput.innerHTML = '';
        if (!data || data.length === 0) {
            resultsOutput.innerHTML = '<p class="placeholder">No entities found in the text.</p>';
            return;
        }

        const fragment = document.createDocumentFragment();
        data.forEach(item => {
            if (item.label === 'O') {
                fragment.appendChild(document.createTextNode(item.text));
            } else {
                const span = document.createElement('span');
                span.className = `entity entity-${item.label}`;
                span.textContent = item.text;
                span.setAttribute('data-label', item.label);
                fragment.appendChild(span);
            }
        });
        resultsOutput.appendChild(fragment);
    };

    // Handle Analyze button click
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze.');
            return;
        }
        
        showLoader(true);
        resultsOutput.innerHTML = '<p class="placeholder">Analyzing...</p>';

        try {
            // **IMPORTANT**: Replace this mock fetch with the real one for your Flask backend.
            // const response = await fetch('/analyze', {
            //     method: 'POST',
            //     headers: {
            //         'Content-Type': 'application/json',
            //     },
            //     body: JSON.stringify({ text: text }),
            // });

            // if (!response.ok) {
            //     throw new Error(`Server error: ${response.statusText}`);
            // }
            // const data = await response.json();
            
            // --- MOCK RESPONSE (for frontend development) ---
            // This part simulates a network request and a response from the Flask server.
            // Replace this with the code above once your backend is ready.
            const data = await new Promise(resolve => {
                setTimeout(() => {
                    // This is an example of the data structure your Flask API should return.
                    // It's a list of objects, where each object is a token.
                    // 'label': 'O' means it's not a named entity.
                    // 'label': 'PER', 'ORG', etc., are the entity types.
                    const sampleText = "Apple is looking at buying U.K. startup for $1 billion in London.";
                    if (text === sampleText) {
                         resolve([
                            { text: 'Apple', label: 'ORG' },
                            { text: ' is looking at buying ', label: 'O' },
                            { text: 'U.K.', label: 'LOC' },
                            { text: ' startup for $1 billion in ', label: 'O' },
                            { text: 'London', label: 'LOC' },
                            { text: '.', label: 'O' }
                        ]);
                    } else {
                        // A simple mock for any other text
                        resolve([
                            { text: 'This is a mocked response for "', label: 'O' },
                            { text: text.substring(0, 20) + '...', label: 'MISC' },
                            { text: '". Please connect to the real Flask backend.', label: 'O' },
                        ]);
                    }
                }, 1500); // Simulate 1.5 second delay
            });
            // --- END OF MOCK RESPONSE ---

            renderResults(data);

        } catch (error) {
            console.error('Error:', error);
            resultsOutput.innerHTML = `<p class="placeholder" style="color: red;">Error analyzing text. Please check the console for details.</p>`;
        } finally {
            showLoader(false);
        }
    });
    
    // Initial setup
    populateLegend();
});
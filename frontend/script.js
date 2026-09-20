const newsText = document.getElementById("newsText");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("resultBox");
const errorMessage = document.getElementById("errorMessage");


// Character counter
newsText.addEventListener("input", function () {
    const count = newsText.value.length;

    charCount.textContent = count + " characters";
});


// Analyze button
analyzeBtn.addEventListener("click", function () {

    const text = newsText.value.trim();

    // Check empty input
    if (text === "") {
        errorMessage.textContent = "Please enter some news content.";
        resultBox.style.display = "none";
        return;
    }

    errorMessage.textContent = "";

    /*
       Temporary result for frontend testing.
       Later this will be replaced by the
       machine learning backend response.
    */

    document.getElementById("prediction").textContent = "Analysis Ready";
    document.getElementById("confidence").textContent = "Waiting";

    document.getElementById("sourceScore").textContent = "--%";
    document.getElementById("languageScore").textContent = "--%";
    document.getElementById("factScore").textContent = "--%";
    document.getElementById("biasScore").textContent = "--%";

    resultBox.style.display = "block";
});
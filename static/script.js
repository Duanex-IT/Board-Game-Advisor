document.addEventListener("DOMContentLoaded", function () {
    let input = document.getElementById("message");

    input.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });
});

async function sendMessage() {
    let input = document.getElementById("message");
    if (input.value.trim() === "") return;

    let messageBox = document.getElementById("chat-box");
    let userMsgDiv = document.createElement("div");
    userMsgDiv.classList.add("message", "sent");
    userMsgDiv.textContent = input.value;
    messageBox.appendChild(userMsgDiv);

    let response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: "123", message: input.value })
    });

    let data = await response.json();
    let botMsgDiv = document.createElement("div");
    botMsgDiv.classList.add("message", "received");
    botMsgDiv.textContent = data.response;
    messageBox.appendChild(botMsgDiv);

    input.value = "";
    messageBox.scrollTop = messageBox.scrollHeight;
}
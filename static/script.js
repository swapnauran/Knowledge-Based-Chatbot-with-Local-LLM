document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    
    // Function to get current time
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // Function to create a message element
    function createMessageElement(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender + '-message');
        
        if (typeof content === 'object' && content.isTyping) {
            messageDiv.classList.add('typing');
            
            const typingIndicator = document.createElement('div');
            typingIndicator.classList.add('typing-indicator');
            
            for (let i = 0; i < 3; i++) {
                const dot = document.createElement('span');
                typingIndicator.appendChild(dot);
            }
            
            messageDiv.appendChild(typingIndicator);
        } else {
            messageDiv.textContent = content;
            
            const timestamp = document.createElement('div');
            timestamp.classList.add('timestamp');
            timestamp.textContent = getCurrentTime();
            messageDiv.appendChild(timestamp);
        }
        
        return messageDiv;
    }
    
    // Function to add a message to the chat
    function addMessage(content, sender) {
        const messageElement = createMessageElement(content, sender);
        chatMessages.appendChild(messageElement);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageElement;
    }
    
    // Function to remove a message element
    function removeMessage(messageElement) {
        chatMessages.removeChild(messageElement);
    }
    
    // Send message function
    function sendMessage() {
        const message = messageInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        messageInput.value = '';
        
        // Show typing indicator
        const typingIndicator = addMessage({ isTyping: true }, 'bot');
        
        // Simulate backend delay and response
        setTimeout(() => {
            // Send to backend
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                // Remove typing indicator
                removeMessage(typingIndicator);
                
                // Add bot response to chat
                addMessage(data.response, 'bot');
            })
            .catch(error => {
                console.error('Error:', error);
                
                // Remove typing indicator
                removeMessage(typingIndicator);
                
                // Add error message
                addMessage('Sorry, I encountered an error. Please try again later.', 'bot');
            });
        }, Math.random() * 1000 + 1000); // Random delay between 1-2 seconds for realistic typing simulation
    }
    
    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Focus input field on load
    messageInput.focus();
    
    // Add some initial animation to show the chatbot is active
    setTimeout(() => {
        const firstMessage = document.querySelector('.bot-message');
        if (firstMessage) {
            firstMessage.style.animation = 'none';
            void firstMessage.offsetWidth; // Trigger reflow
            firstMessage.style.animation = 'fadeIn 0.5s ease';
        }
    }, 500);
});
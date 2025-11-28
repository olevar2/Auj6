from typing import Any


class MessagingService:
    def send_message(self, message: str):
        print(f"Message sent: {message}")
    
    def publish(self, topic: str, message: Any):
        print(f"Published to {topic}: {message}")

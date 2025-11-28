from auj_platform.src.core.messaging_service import MessagingService

class MessagingServiceFactory:
    @staticmethod
    def create_messaging_service() -> MessagingService:
        return MessagingService()

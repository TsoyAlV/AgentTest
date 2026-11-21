class ServiceError(Exception):
    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


class CitationError(ServiceError):
    pass


class IncompleteModelOutput(ServiceError):
    pass

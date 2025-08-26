class SecurityHeadersMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        response["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response["Cross-Origin-Embedder-Policy"] = "credentialless"
        response["Cross-Origin-Opener-Policy"] = "same-origin"
        response["Cross-Origin-Resource-Policy"] = "same-origin"

        return response

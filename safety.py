import asyncio

class RequestLimits:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        status = None
        try:
            async with asyncio.timeout(10):
                for _ in range(4096):
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > 65536:
                        status = 413
                        break
                    if not message.get("more_body", False):
                        break
                else:
                    status = 413
        except TimeoutError:
            status = 408
        if status:
            await send({"type": "http.response.start", "status": status, "headers": []})
            return await send({"type": "http.response.body", "body": b"Request limit exceeded"})
        consumed = False
        async def bounded_receive():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}
        await self.app(scope, bounded_receive, send)

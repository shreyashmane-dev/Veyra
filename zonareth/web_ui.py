"""
ZONARETH ChatGPT-4 Web Studio
"""

from veyra.web_ui import launch_server

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Zonareth ChatGPT-4 Web Studio")
    parser.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    launch_server(port=args.port, open_browser=not args.no_browser)

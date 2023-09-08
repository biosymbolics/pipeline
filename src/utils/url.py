"""
Utils related to urls
"""
import regex as re


def url_to_filename(url: str) -> str:
    """Turns a URL into a valid filename.

    Args:
      url: The URL to be converted.

    Returns:
      The valid filename.
    """

    # Remove the protocol from the URL.
    filename = re.sub(r"^https?://", "", url)

    # Replace all invalid characters in the URL with underscores.
    filename = re.sub(r"[^\w\-.]", "", filename)

    # Strip any leading or trailing underscores.
    filename = filename.strip("_")

    return filename

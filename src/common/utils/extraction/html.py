from bs4 import BeautifulSoup


def strip_inline_styles(html: str) -> str:
    """
    Strip inline styles from html

    Args:
        html (str): the html
    """
    soup = BeautifulSoup(html, "html.parser")

    # Create a new BeautifulSoup object with the desired changes
    [element.attrs.pop("style", None) for element in soup.select("[style]")]

    # Return the modified HTML
    return str(soup)

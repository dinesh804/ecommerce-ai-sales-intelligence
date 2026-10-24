import requests
from bs4 import BeautifulSoup


def print_secret_message(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    points = []

    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])

        if len(cells) < 3:
            continue

        values = [cell.get_text(strip=True) for cell in cells]

        try:
            x = int(values[0])
            char = values[1]
            y = int(values[2])
        except (ValueError, IndexError):
            continue

        points.append((x, y, char))

    if not points:
        raise ValueError("No data found in the document.")

    max_x = max(x for x, y, char in points)
    max_y = max(y for x, y, char in points)

    grid = [
        [" "] * (max_x + 1)
        for _ in range(max_y + 1)
    ]

    for x, y, char in points:
        grid[y][x] = char

    for row in grid:
        print("".join(row))


# Test the function
url = "https://docs.google.com/document/d/e/2PACX-1vSvM5gDlNvt7npYHhp_XfsJvuntUhq184By5xO_pA4b_gCWeXb6dM6ZxwN8rE6S4ghUsCj2VKR21oEP/pub"

print_secret_message(url)
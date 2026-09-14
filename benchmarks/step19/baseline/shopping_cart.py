"""A small shopping cart built on the pricing module."""

from pricing import line_total


class ShoppingCart:
    def __init__(self):
        self._items = []

    def add_item(self, name: str, quantity: int, unit_price: float) -> None:
        self._items.append({"name": name, "quantity": quantity, "price": unit_price})

    def item_count(self) -> int:
        return sum(item["quantity"] for item in self._items)

    def total(self) -> float:
        return round(
            sum(
                line_total(item["price"], item["quantity"])
                for item in self._items
            ),
            2,
        )
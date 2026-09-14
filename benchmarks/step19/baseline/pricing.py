"""Pricing rules for shopping-cart line items."""

BULK_QUANTITY = 10
BULK_DISCOUNT = 0.9


def line_total(quantity: int, unit_price: float) -> float:
    """Total price for `quantity` units, discounting 10% at BULK_QUANTITY."""
    gross = quantity * unit_price
    if quantity >= BULK_QUANTITY:
        gross *= BULK_DISCOUNT
    return round(gross, 2)
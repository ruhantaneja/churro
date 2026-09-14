from pricing import BULK_DISCOUNT, BULK_QUANTITY, line_total


def test_normal_price_small_quantity():
    assert line_total(1, 2.5) == 2.5


def test_normal_price_multiple_units():
    assert line_total(3, 2.0) == 6.0


def test_expensive_unit_small_quantity_is_not_discounted():
    assert line_total(2, 15.0) == 30.0


def test_bulk_discount_applies_at_threshold():
    assert line_total(BULK_QUANTITY, 5.0) == 45.0


def test_bulk_discount_applies_above_threshold():
    assert line_total(20, 3.0) == 54.0


def test_bulk_discount_keys_on_quantity_not_unit_price():
    assert line_total(2, 15.0) == 30.0
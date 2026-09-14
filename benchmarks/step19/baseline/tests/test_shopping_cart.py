from shopping_cart import ShoppingCart


def test_empty_cart_total_is_zero():
    assert ShoppingCart().total() == 0.0


def test_single_small_item():
    cart = ShoppingCart()
    cart.add_item("apple", 2, 1.5)
    assert cart.total() == 3.0
    assert cart.item_count() == 2


def test_multiple_small_items():
    cart = ShoppingCart()
    cart.add_item("apple", 2, 1.5)
    cart.add_item("banana", 3, 0.5)
    assert cart.total() == 4.5


def test_bulk_quantity_gets_discount_through_cart():
    cart = ShoppingCart()
    cart.add_item("soap", 10, 4.0)
    assert cart.total() == 36.0


def test_mixed_cart_discounts_only_bulk_items():
    cart = ShoppingCart()
    cart.add_item("apple", 2, 1.5)
    cart.add_item("soap", 10, 4.0)
    assert cart.total() == 39.0


def test_bulk_discount_does_not_depend_on_unit_price():
    cart = ShoppingCart()
    cart.add_item("soap", 12, 2.5)
    assert cart.total() == 27.0
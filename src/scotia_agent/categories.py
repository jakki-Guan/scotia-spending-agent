"""
Keyword-based transaction categorizer for Scotia Visa transactions.

Derived from Jake's real Scene Visa data (1000 txns, 267 unique merchants).
Order matters: more specific patterns must come before generic ones.
Match is substring-based on the lowercased Description field.

Usage:
    from categories import categorize
    cat = categorize("uber canada/ubereats")  # -> "food_delivery"

Anything that doesn't match returns "uncategorized" — those rows should
fall through to the LLM categorizer (the hybrid path in your DESIGN.md).
"""

from __future__ import annotations

# (substring, category) — checked top-to-bottom, first match wins.
RULES: list[tuple[str, str]] = [
    # ---------- Bank / fees / internal transfers (check FIRST: very specific) ----------
    ("scotia credit card protec", "bank_fees"),
    ("scotia sccp premium", "bank_fees"),
    ("interest charges", "bank_fees"),
    ("scotiabank transit", "bank_fees"),  # branch txn / adjustment
    ("payment from", "payment"),  # credit card payment IN
    ("from - *****", "payment"),
    ("manulife travel insurance", "insurance"),
    ("immigration canada online", "government_fees"),
    # ---------- Subscriptions (digital) ----------
    ("claude.ai", "subscription_ai"),
    ("anthropic", "subscription_ai"),
    ("openai", "subscription_ai"),
    (
        "amzn mktp",
        "subscription_shopping_online",
    ),  # Amazon subscription fallback (e.g. Kindle Unlimited)
    ("chatgpt", "subscription_ai"),
    ("spotify", "subscription_media"),
    ("google *google one", "subscription_cloud"),
    ("google one", "subscription_cloud"),
    ("amazon.ca prime", "subscription_media"),
    ("amazon prime", "subscription_media"),
    ("linkedin", "subscription_pro"),
    # ---------- Telecom / utilities ----------
    ("rogers", "telecom"),
    ("bell canada", "telecom"),
    ("wyse meter", "utilities"),
    ("paymentus", "utilities"),
    # ---------- Insurance / government / fees ----------
    ("max insurance", "insurance"),
    ("fg hamilton police", "government_fees"),
    ("precise parklink", "parking"),
    ("canada wide parking", "parking"),
    # ---------- Education ----------
    ("queen's", "education"),
    ("parchment", "education"),
    ("vue*testing", "education"),
    ("xyna international", "education"),
    ("wl *vue", "education"),
    # ---------- Transport: rideshare (Uber Eats BEFORE Uber!) ----------
    ("ubereats", "food_delivery"),
    ("uber* eats", "food_delivery"),
    ("uber*eats", "food_delivery"),
    ("uber eats", "food_delivery"),
    ("ubertrip", "rideshare"),
    ("uber* trip", "rideshare"),
    ("uber *trip", "rideshare"),
    ("uber holdings", "rideshare"),
    ("uber canada", "rideshare"),  # generic Uber fallback
    ("uber", "rideshare"),
    # ---------- Transport: transit / car-share / travel ----------
    ("presto", "transit"),
    ("go tvm", "transit"),
    ("hamilton - go", "transit"),
    ("via rail", "transit"),
    ("pioneer stn", "fuel"),
    ("cannon st esso", "fuel"),
    ("esso", "fuel"),
    ("petro-canada", "fuel"),
    ("1000503499 ontario", "fuel"),
    ("communauto", "car_rental"),
    ("enterprise canada", "car_rental"),
    ("poparide", "rideshare"),
    ("booking.com", "travel"),
    ("cn tower online ticket", "travel"),
    ("blue mountain", "travel"),
    # ---------- Groceries & convenience ----------
    ("james street market", "groceries"),  # 143 hits — main grocery
    ("big bee food mart", "groceries"),
    ("big bear food mart", "groceries"),
    ("jestar convience", "groceries"),
    ("walmart supercenter", "groceries"),
    ("wal-mart", "groceries"),
    ("walmart.ca", "shopping_online"),
    ("dollarama", "shopping_retail"),
    ("home hardware", "shopping_retail"),
    ("shoppers drug mart", "pharmacy"),
    ("sdm ", "pharmacy"),
    # ---------- Alcohol ----------
    ("lcbo", "alcohol"),
    # ---------- Coffee / tea / bubble tea ----------
    ("tim hortons", "coffee"),
    ("timshop.ca", "coffee"),
    ("starbucks", "coffee"),
    ("mcdonald", "fast_food"),
    ("kfc", "fast_food"),
    ("taco bell", "fast_food"),
    ("mary browns", "fast_food"),
    ("domino", "fast_food"),
    ("popeye", "fast_food"),
    ("a&w", "fast_food"),
    ("subway", "fast_food"),
    ("clucker", "fast_food"),
    # bubble tea / dessert
    ("coco hamilton", "bubble_tea"),
    ("coco bubble tea", "bubble_tea"),
    ("hero tea", "bubble_tea"),
    ("gong cha", "bubble_tea"),
    ("ten ren", "bubble_tea"),
    ("micha milktea", "bubble_tea"),
    ("daigyo cafe", "bubble_tea"),
    ("charlie s tea", "bubble_tea"),
    ("big scoops", "dessert"),
    ("budapest bakeshop", "dessert"),
    # ---------- Restaurants (Asian — your top category by variety) ----------
    ("menya kyu", "restaurant"),
    ("tondou ramen", "restaurant"),
    ("sapporo cuisine", "restaurant"),
    ("niku niku", "restaurant"),
    ("aichi japan", "restaurant"),
    ("liuyishou", "restaurant"),
    ("porcelain hotpot", "restaurant"),
    ("szechuan noodle", "restaurant"),
    ("shi miaodao", "restaurant"),
    ("noodle & dumplings", "restaurant"),
    ("pho anh vu", "restaurant"),
    ("xe kem", "restaurant"),
    ("saltlick smokehouse", "restaurant"),
    ("flora pizzeria", "restaurant"),
    ("grill shack", "restaurant"),
    ("ciao bella", "restaurant"),
    ("mars village", "restaurant"),
    ("los mayas", "restaurant"),
    ("piazzetta st-jean", "restaurant"),
    ("the works", "restaurant"),
    ("chef bai", "restaurant"),
    ("secco", "restaurant"),
    ("jan bingo", "restaurant"),
    ("ramen isshin", "restaurant"),
    ("bouillon bilk", "restaurant"),
    ("l'entrecote", "restaurant"),
    ("tst-hexagon", "restaurant"),
    ("tst-the standard", "restaurant"),
    ("goldies fast food", "restaurant"),
    # ---------- Personal care ----------
    ("family hair cut", "personal_care"),
    ("relx qlab", "vape"),
    # ---------- Entertainment / fitness / hobbies ----------
    ("ticketmaster", "entertainment"),
    ("coca-cola coliseum", "entertainment"),
    ("gravity climbing", "fitness"),
    ("jump +", "fitness"),
    ("bungie store", "gaming"),
    ("xsolla", "gaming"),
    ("iris galerie", "entertainment"),
    ("4 u bad centre", "fitness"),
    ("apple.com/bill", "gaming"),
    ("paypal *youzusingap", "gaming"),  # gaming
    # ---------- Online shopping ----------
    ("amazon.ca", "shopping_online"),
    ("amazon*", "shopping_online"),  # generic Amazon fallback (after specific Amazon rules above)
    ("best buy", "shopping_online"),
    ("canada computers", "shopping_online"),
    ("tri-star computer", "shopping_online"),
    ("taobao", "shopping_online"),
    ("shoe point", "shopping_retail"),
    ("styledemocracy", "shopping_online"),
    ("octobers very own", "shopping_online"),
    ("arcteryx", "shopping_online"),
    ("jellycat inc", "shopping_online"),
    ("uniqlo", "shopping_online"),
    ("toronto duty free", "shopping_online"),
    ("niagara outlet", "shopping_online"),
    # ---------- Coca-Cola bottling = vending/workplace, not a drink purchase ----------
    ("coca cola", "vending"),
    ("coca-cola", "vending"),
    # ---------- Catch-all PayPal (after specific PayPal rules above) ----------
    ("paypal", "shopping_online"),
]


def categorize(description: str) -> str:
    """Return the category for a transaction description, or 'uncategorized'."""
    if not description:
        return "uncategorized"
    desc = description.lower().strip()
    for needle, cat in RULES:
        if needle in desc:
            return cat
    return "uncategorized"

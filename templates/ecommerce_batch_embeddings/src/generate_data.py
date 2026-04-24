"""
Synthetic e-commerce product catalog generator.
Writes Parquet to /mnt/cluster_storage/ecommerce-demo/raw/products.parquet
"""
import argparse
import os
import random
import uuid

import pandas as pd
from faker import Faker

fake = Faker()

SCALE_MAP = {"small": 1_000, "medium": 100_000, "large": 2_000_000}

CATALOG = {
    "electronics": {
        "brands": ["Sony", "Samsung", "Apple", "Bose", "JBL", "Anker", "Logitech", "LG", "Jabra", "Belkin"],
        "templates": [
            ("{brand} Wireless Bluetooth Headphones",
             "Premium over-ear noise cancelling headphones with {hours}-hour battery life, "
             "Hi-Res Audio certification, and foldable design for travel. Features {brand} LDAC codec "
             "for wireless high-resolution audio streaming. Includes carrying case and audio cable."),
            ("{brand} True Wireless Earbuds",
             "Compact true wireless earbuds with active noise cancellation, {hours}-hour total playback, "
             "IPX4 water resistance, and touch controls. Charging case provides 3 additional charges. "
             "Compatible with voice assistants."),
            ("{brand} USB-C Fast Charger {watts}W",
             "GaN technology fast charger supports USB Power Delivery up to {watts}W. "
             "Charges laptops, tablets, and smartphones simultaneously. Compact design with foldable plug. "
             "Compatible with all USB-C devices."),
            ("{brand} Portable Bluetooth Speaker",
             "360-degree surround sound speaker with IP67 waterproof and dustproof rating. "
             "{hours}-hour playtime, built-in microphone for hands-free calls, and USB-C charging. "
             "Pairs two speakers for stereo sound."),
            ("{brand} Mechanical Keyboard",
             "Compact tenkeyless mechanical keyboard with {switch} switches, per-key RGB backlighting, "
             "and USB-C detachable cable. N-key rollover for gaming. Available in multiple switch types. "
             "Aluminum top frame with PBT keycaps."),
        ],
    },
    "clothing": {
        "brands": ["Nike", "Adidas", "Levi's", "H&M", "Zara", "Patagonia", "Under Armour", "Gap", "Uniqlo", "Carhartt"],
        "templates": [
            ("{brand} Men's Running Shoes",
             "Lightweight performance running shoe with responsive foam midsole and breathable mesh upper. "
             "Heel drop {drop}mm, weight {weight}oz per shoe. Suitable for road running and gym workouts. "
             "Reinforced heel counter for stability. Available in wide widths."),
            ("{brand} Women's Athletic Leggings",
             "High-waist compression leggings with moisture-wicking fabric and hidden waistband pocket. "
             "Four-way stretch material provides unrestricted movement for yoga, running, and training. "
             "Flatlock seams prevent chafing. UPF 50+ sun protection."),
            ("{brand} Insulated Winter Jacket",
             "600-fill-power down insulated jacket with wind and water resistant shell. "
             "Packable into its own chest pocket. Elastic hem and cuffs seal in warmth. "
             "Two hand pockets and one internal pocket. Ideal for hiking, skiing, and everyday wear."),
            ("{brand} Classic Straight Jeans",
             "Mid-rise straight-fit jeans in {wash} wash denim. Made from {cotton}% cotton denim "
             "with slight stretch for comfort. Five-pocket styling with button fly. "
             "Machine washable. Available in multiple inseam lengths."),
            ("{brand} Performance Polo Shirt",
             "Moisture-wicking polo shirt with UV protection for golf and outdoor activities. "
             "Stretch fabric allows full range of motion. Self-fabric collar stays crisp all day. "
             "Three-button placket and side vents for ventilation."),
        ],
    },
    "home_goods": {
        "brands": ["Cuisinart", "KitchenAid", "Dyson", "Instant Pot", "Ninja", "OXO", "Lodge", "Vitamix", "Breville", "Calphalon"],
        "templates": [
            ("{brand} Air Fryer {quarts}Qt",
             "Digital air fryer with {quarts}-quart basket capacity. {temp_range} temperature range. "
             "7 preset cooking functions including air fry, roast, bake, and dehydrate. "
             "Dishwasher-safe basket and crisper plate. Non-stick interior for easy cleaning."),
            ("{brand} Stand Mixer {watts}W",
             "Professional-grade {watts}-watt stand mixer with {bowl}Qt stainless steel bowl. "
             "10-speed settings with soft start to prevent flour clouds. Includes flat beater, "
             "dough hook, and wire whip. Planetary mixing action reaches every part of bowl."),
            ("{brand} Robot Vacuum",
             "Smart robot vacuum with {suction}Pa suction power, LiDAR navigation, and multi-floor mapping. "
             "Compatible with Alexa and Google Home. Auto-empty base holds up to 60 days of debris. "
             "Works on carpets and hard floors. App control with scheduling."),
            ("{brand} Cast Iron Skillet {inch}\"",
             "Pre-seasoned cast iron skillet perfect for stovetop, oven, grill, and campfire cooking. "
             "{inch}-inch cooking surface. Superior heat retention and even heat distribution. "
             "Naturally non-stick surface improves with use. Made in USA."),
            ("{brand} Cordless Stick Vacuum",
             "Lightweight cordless vacuum with {suction}AW suction and {runtime}-minute runtime on single charge. "
             "Converts to handheld for stairs and car cleaning. Anti-tangle brush roll for long hair. "
             "Wall-mounted docking station included."),
        ],
    },
    "beauty": {
        "brands": ["CeraVe", "Neutrogena", "L'Oréal", "Olay", "The Ordinary", "Paula's Choice", "Cetaphil", "Aveeno", "Drunk Elephant", "La Roche-Posay"],
        "templates": [
            ("{brand} Daily Moisturizing Face Cream SPF {spf}",
             "Lightweight daily moisturizer with broad-spectrum SPF {spf} protection. "
             "Formulated with hyaluronic acid and ceramides to restore skin's natural barrier. "
             "Oil-free, non-comedogenic, and fragrance-free. Suitable for sensitive skin. "
             "Dermatologist tested."),
            ("{brand} Vitamin C Serum",
             "Brightening vitamin C serum with {percent}% L-ascorbic acid, vitamin E, and ferulic acid. "
             "Reduces appearance of dark spots and uneven skin tone. Antioxidant protection against "
             "environmental damage. Apply before SPF for enhanced protection. Suitable for all skin types."),
            ("{brand} Retinol Night Cream",
             "Anti-aging night cream with {percent}% retinol and peptide complex. Reduces appearance of "
             "fine lines and wrinkles overnight. Hyaluronic acid provides intense hydration. "
             "Fragrance-free and dermatologist tested. Start with 2-3 nights per week."),
            ("{brand} Hydrating Face Wash",
             "Gentle hydrating cleanser that removes makeup, dirt, and excess oil without disrupting "
             "the skin barrier. Contains ceramides and niacinamide. pH-balanced formula suitable for "
             "dry to normal skin. Non-foaming texture. Use morning and evening."),
            ("{brand} Eye Cream",
             "Concentrated eye cream targeting dark circles, puffiness, and fine lines. "
             "Caffeine and peptide complex reduces visible signs of fatigue. Lightweight gel-cream "
             "absorbs quickly. Fragrance-free and ophthalmologist tested. Use AM and PM."),
        ],
    },
    "sports": {
        "brands": ["Wilson", "Callaway", "Titleist", "Yeti", "Hydro Flask", "CamelBak", "Black Diamond", "Osprey", "TRX", "Bowflex"],
        "templates": [
            ("{brand} Adjustable Dumbbell Set",
             "Space-saving adjustable dumbbells replace {count} individual weights. "
             "Dial-select weight system adjusts from {min}lbs to {max}lbs in 5lb increments. "
             "Ergonomic handle with comfortable grip. Storage tray included. "
             "Ideal for home gyms with limited space."),
            ("{brand} Insulated Water Bottle {oz}oz",
             "Double-wall vacuum insulated stainless steel water bottle keeps drinks cold {cold} hours "
             "and hot {hot} hours. BPA-free and dishwasher safe. Wide mouth for easy filling and ice. "
             "Leak-proof lid. {oz}oz capacity. Available in multiple colors."),
            ("{brand} Yoga Mat {mm}mm",
             "Premium {mm}mm thick yoga mat with non-slip textured surface on both sides. "
             "Closed-cell surface prevents moisture and odor absorption. "
             "Alignment lines for proper positioning. Includes carrying strap. "
             "Made from eco-friendly TPE material."),
            ("{brand} Hiking Backpack {liters}L",
             "{liters}-liter hiking backpack with aluminum frame and padded hip belt for load transfer. "
             "Hydration-compatible with 3L reservoir sleeve. Multiple external attachment points "
             "for trekking poles and sleeping pad. Rain cover included. Fits torso lengths {min}\"-{max}\"."),
            ("{brand} Resistance Band Set",
             "Set of {count} resistance bands ranging from {min}lbs to {max}lbs resistance. "
             "Made from natural latex with anti-snap safety layer. Includes handles, ankle straps, "
             "door anchor, and carrying bag. Suitable for physical therapy and strength training."),
        ],
    },
}

FILL_VALUES = {
    "hours": lambda: random.choice([20, 24, 30, 36, 40]),
    "watts": lambda: random.choice([45, 65, 90, 100, 140]),
    "switch": lambda: random.choice(["Cherry MX Red", "Cherry MX Blue", "Gateron Brown", "Kailh Speed Silver"]),
    "drop": lambda: random.choice([4, 6, 8, 10, 12]),
    "weight": lambda: round(random.uniform(7.5, 11.5), 1),
    "wash": lambda: random.choice(["light", "medium", "dark", "black", "indigo"]),
    "cotton": lambda: random.choice([98, 99, 100]),
    "quarts": lambda: random.choice([4, 5, 6, 8]),
    "temp_range": lambda: random.choice(["180°F–400°F", "200°F–450°F"]),
    "bowl": lambda: random.choice([5, 6, 7]),
    "suction": lambda: random.choice([2200, 2500, 3000, 4000]),
    "runtime": lambda: random.choice([40, 45, 60]),
    "inch": lambda: random.choice([10, 12]),
    "spf": lambda: random.choice([30, 50]),
    "percent": lambda: round(random.uniform(0.025, 0.3), 3),
    "count": lambda: random.choice([15, 25, 50]),
    "min": lambda: random.choice([5, 10]),
    "max": lambda: random.choice([50, 52, 55]),
    "cold": lambda: random.choice([24, 48]),
    "hot": lambda: random.choice([12, 24]),
    "oz": lambda: random.choice([18, 24, 32, 40]),
    "mm": lambda: random.choice([4, 5, 6]),
    "liters": lambda: random.choice([28, 38, 50, 65]),
}

PRICE_RANGES = {
    "electronics": (19.99, 499.99),
    "clothing": (14.99, 249.99),
    "home_goods": (24.99, 699.99),
    "beauty": (8.99, 129.99),
    "sports": (12.99, 399.99),
}


def _render_template(template: str, brand: str) -> str:
    result = template.replace("{brand}", brand)
    for key, fn in FILL_VALUES.items():
        if f"{{{key}}}" in result:
            result = result.replace(f"{{{key}}}", str(fn()), 1)
    return result


def generate_product(category: str) -> dict:
    config = CATALOG[category]
    brand = random.choice(config["brands"])
    title_tmpl, desc_tmpl = random.choice(config["templates"])
    title = _render_template(title_tmpl, brand)
    description = _render_template(desc_tmpl, brand)
    lo, hi = PRICE_RANGES[category]
    return {
        "product_id": str(uuid.uuid4()),
        "title": title,
        "description": description,
        "category": category,
        "brand": brand,
        "price": round(random.uniform(lo, hi), 2),
    }


def generate_catalog(num_products: int, output_path: str) -> str:
    print(f"Generating {num_products:,} synthetic products...")
    categories = list(CATALOG.keys())
    records = [generate_product(random.choice(categories)) for _ in range(num_products)]
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved {len(df):,} products → {output_path}")
    print(f"  Category breakdown:\n{df['category'].value_counts().to_string()}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--output", default="/mnt/cluster_storage/ecommerce-demo/raw/products.parquet")
    args = parser.parse_args()
    generate_catalog(SCALE_MAP[args.scale], args.output)

def P(x, y, cat): return {"xy": (x, y), "cat": cat}

NAMED_POINTS = {
    # water_zone
    "water_slide":          P(22.82,  0.47, "water"),
    "splash_battle":        P(16.12,  6.94, "water"),
    "wave_pool":            P(10.35,  4.08, "water"),
    "shower_changing_room":        P( 5.81,  0.08, "water"),
    "lazy_river":           P(10.94, -4.60, "water"),
    "log_flume":            P(16.80, -7.46, "water"),

    # entertainment_zone
    "circus_tent":          P(15.61,  -8.15, "show"),
    "musical_fountain":     P(22.62, -16.76, "show"),
    "magic_show":           P(12.92, -22.76, "show"),
    "street_parade":        P( 3.87, -14.21, "show"),
    "live_stage":           P( 4.03,  -4.93, "show"),

    # food_zone
    "snack_bar":            P( 2.48, -13.55, "food"),
    "pizza_plaza":          P(-0.58, -22.64, "food"),
    "food_court":           P(-8.08, -22.65, "food"),
    "ice_cream_kiosk":      P(-12.58,-16.17, "food"),
    "smoothie_station":     P(-7.12,  -9.05, "food"),

    # family_zone
    "train_adventure":      P(-8.55, -10.27, "family"),
    "animal_safari_ride":   P(-14.75,-18.11, "family"),
    "mini_ferris_wheel":    P(-23.08,-13.04, "family"),
    "ball_pit_dome":        P(-17.31, -0.32, "family"),
    "bumper_cars":          P(-10.47, -0.08, "family"),

    # thrill_zone
    "haunted_mine_train":   P(-1.39,   6.31, "thrill"),
    "drop_tower":           P( 3.10,  14.70, "thrill"),
    "roller_coaster":       P(-6.05,  23.08, "thrill"),
    "freefall_cannon":      P(-12.41, 16.19, "thrill"),
    "spinning_vortex":      P(-8.14,  10.70, "thrill"),

    # shopping_zone
    "toy_world":            P( 3.44,  13.34, "shop"),
    "gift_emporium":        P(14.10,  22.85, "shop"),
    "photo_booth":          P(22.91,  16.53, "shop"),
    "candy_store":          P(17.64,   8.74, "shop"),
    "souvenir_shop":        P(10.51,   5.17, "shop"),

    # relaxation
    "relaxation_garden":    P(-12.18, 14.67, "relax"),
    "shaded_benches":       P(-23.20, 12.30, "relax"),
    "quiet_lake_view":      P(-12.18, 14.67, "relax"),
    "zen_courtyard":        P(-23.20, 12.30, "relax"),
    "sky_deck":             P(-12.18, 14.67, "relax"),
}

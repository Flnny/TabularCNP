#Features
poi_features = ['education', 'business', 'retail', 'accommodation', 'culture', 'healthcare',
                'bridge', 'cinema', 'park', 'entertainment', 'religious', 'food', 'parking',
                'transportation', 'warehouse', 'industrial', 'residential', 'construction',
                'market', 'campsite', 'sports', 'public services', 'auto services',
                'financial services', 'boat services', 'farm facilities']

way_features = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "track",
    "footway",
    "path",
    "cycleway",
    "pedestrian"
]

Prosperity= ['Prosperity']

dense_features = [
    'longitude', 'latitude', 'building_area', 'total_floors', 'listing_price', 'year_built',
    'price_adjustment', 'viewing_count',
    'followers_count', 'visitors_count', 'ladders', 'houses',
    'kitchens', 'bedrooms', 'living_rooms', 'bathrooms',
    'listing_year', 'listing_month', 'listing_day', 'nei_price'
]
sparse_features = [
    'district_name', 'business_area', 'community_name',
    'apartment_type', 'building_type', 'house_orientation',
    'renovation_status', 'building_structure',
    'heating_method', 'elevator_available', 'transaction_rights',
    'house_usage', 'house_age', 'house_ownership'
]
target = ['avg_transaction']
dense_features = dense_features + poi_features + way_features + Prosperity
features = dense_features + sparse_features
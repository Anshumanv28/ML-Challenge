# Mapping of entity types to their respective allowed units
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {
        'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'
    },
    'maximum_weight_recommendation': {
        'gram',
        'kilogram',
        'microgram',
        'milligram',
        'ounce',
        'pound',
        'ton'
    },
    'voltage': {
        'kilovolt',
        'millivolt',
        'volt'
    },
    'wattage': {
        'kilowatt',
        'watt'
    },
    'item_volume': {
        'centilitre',
        'cubic foot',
        'cubic inch',
        'cup',
        'decilitre',
        'fluid ounce',
        'gallon',
        'imperial gallon',
        'litre',
        'microlitre',
        'millilitre',
        'pint',
        'quart'
    }
}

# Set of all allowed units extracted from the entity_unit_map
allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

# Debugging: Print allowed units to ensure correct extraction
if __name__ == "__main__":
    print("Allowed units:", allowed_units)

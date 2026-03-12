"""DIGIT Random Prompt Generator — builds randomized cinematic image prompts."""

import random


# --- Category Data ---

SETTING = [
    "interior", "exterior",
]

LOCATIONS = [
    "abandoned warehouse", "rooftop bar at sunset", "neon-lit Tokyo alley",
    "Moroccan riad courtyard", "brutalist parking garage", "tropical greenhouse",
    "midcentury modern living room", "underground subway platform", "foggy pier",
    "Italian piazza at dawn", "desert highway gas station", "Icelandic black sand beach",
    "Victorian library", "industrial loft", "rain-soaked Parisian street",
    "bamboo forest path", "art deco hotel lobby", "concrete skatepark",
    "Mediterranean cliffside villa", "smoky jazz club", "snowy mountain cabin",
    "airport terminal at 3am", "overgrown ruins", "chrome-and-glass penthouse",
    "fishing village harbor", "cathedral interior", "graffiti-covered tunnel",
    "sun-bleached motel pool", "floating market", "volcanic landscape",
    "old train station", "botanical garden", "underground wine cellar",
    "rooftop garden", "abandoned theater", "seaside boardwalk",
    "misty Scottish highlands", "Hong Kong night market", "Havana street corner",
    "minimalist gallery space", "rustic barn", "coral reef underwater",
    "space station observation deck", "ancient temple", "urban fire escape",
    "lavender field in Provence", "dusty Western saloon", "frozen lake",
    "neon-lit diner", "marble palace hallway", "dense jungle clearing",
    "Brooklyn brownstone stoop", "Norwegian fjord", "carnival midway",
    "sunflower field", "derelict factory floor", "cliffside lighthouse",
    "Chinatown back alley", "vintage record shop", "moonlit cemetery",
    "alpine meadow", "underground bunker", "Venice canal",
    "Pacific Coast Highway overlook", "Saharan sand dunes", "London pub interior",
    "cherry blossom park", "Miami art deco district", "Scottish castle courtyard",
    "Balinese rice terraces", "New York City subway car", "Greek island whitewash village",
    "redwood forest", "Marrakech souk", "Berlin techno club",
    "Route 66 roadside motel", "Japanese zen garden", "São Paulo favela rooftop",
    "Swiss chalet balcony", "abandoned amusement park", "mangrove swamp",
    "Santorini caldera view", "Mexico City mercado", "Helsinki sauna",
    "Patagonian glacier", "Nashville honky-tonk bar", "Rajasthani desert fort",
    "Amsterdam canal house", "Australian outback", "Seoul street food alley",
]

TIME_OF_DAY = [
    "golden hour", "blue hour", "high noon", "dawn", "dusk",
    "midnight", "late afternoon", "early morning", "twilight", "predawn",
    "magic hour", "civil twilight", "nautical twilight", "solar noon",
    "overcast midday", "first light", "last light", "witching hour",
    "sunrise", "sunset", "mid-morning", "late evening",
]

WEATHER = [
    "clear sky", "overcast", "heavy rain", "light drizzle", "dense fog",
    "snowfall", "thunderstorm", "misty", "partly cloudy", "hazy sunshine",
    "blizzard", "dust storm", "humid and still", "wind-swept", "freezing rain",
    "rainbow after storm", "heat shimmer", "gentle breeze", "dramatic clouds",
    "aurora borealis", "volcanic ash haze", "monsoon downpour", "frost",
    "sleet", "balmy and tropical", "dry desert heat", "ocean spray mist",
    "rolling thunder clouds", "crisp autumn air", "sweltering summer heat",
]

MOOD = [
    "cinematic and moody", "warm and nostalgic", "cold and isolating",
    "dreamy and ethereal", "gritty and raw", "serene and peaceful",
    "tense and suspenseful", "romantic and intimate", "melancholic",
    "euphoric and vibrant", "eerie and unsettling", "contemplative",
    "chaotic and energetic", "solemn and reverent", "playful and whimsical",
    "dark and brooding", "hopeful and uplifting", "mysterious",
    "luxurious and decadent", "minimalist and clean", "apocalyptic",
    "psychedelic", "noir", "pastoral and idyllic", "fierce and powerful",
    "delicate and fragile", "bold and confrontational", "quiet and still",
    "electric and alive", "bittersweet", "otherworldly", "vintage and faded",
    "hyper-saturated pop art", "wabi-sabi imperfection", "grand and epic",
]

CAMERA = [
    "ARRI Alexa 65", "RED V-Raptor", "Sony Venice 2", "Canon C500 Mark II",
    "Panavision Millennium DXL2", "ARRI Alexa Mini LF", "Blackmagic URSA Mini Pro 12K",
    "RED Komodo", "Sony FX9", "Canon EOS R5 C", "Hasselblad X2D 100C",
    "Leica SL3", "Nikon Z9", "Fujifilm GFX 100 II", "Phase One XF IQ4 150MP",
    "IMAX MSM 9802", "Bolex H16", "Super 8mm", "16mm Éclair NPR",
    "35mm Panavision Panaflex", "65mm IMAX", "anamorphic Panavision C-Series",
    "Mamiya RZ67", "Pentax 67", "large format 8x10 view camera",
    "medium format Rolleiflex", "pinhole camera", "drone DJI Inspire 3",
    "GoPro Hero 12", "iPhone 16 Pro Max", "disposable Kodak FunSaver",
    "Polaroid SX-70", "Lomo LC-A", "Canon AE-1", "Nikon F3",
    "Contax T2", "Yashica T4", "Minolta X-700",
]

LENS = [
    "24mm wide angle", "35mm prime", "50mm f/1.2", "85mm portrait",
    "135mm telephoto", "200mm f/2", "14mm ultra-wide", "28mm street",
    "40mm pancake", "70-200mm zoom", "100mm macro", "300mm super-telephoto",
    "16mm fisheye", "45mm tilt-shift", "Petzval 85mm swirly bokeh",
    "anamorphic 40mm", "anamorphic 75mm", "Lensbaby Velvet 56",
    "vintage Helios 44-2 58mm", "Cooke Speed Panchro", "Zeiss Super Speed",
    "Leica Summilux 50mm", "Canon K35 prime set", "Kowa anamorphic",
]

FILM_STOCK = [
    "Kodak Portra 400", "Kodak Portra 800", "Kodak Ektar 100",
    "Fujifilm Pro 400H", "Kodak Tri-X 400", "Ilford HP5 Plus",
    "Kodak Gold 200", "Fujifilm Velvia 50", "Fujifilm Superia 400",
    "CineStill 800T", "Kodak Vision3 500T", "Kodak Vision3 250D",
    "Kodak Vision3 50D", "Ilford Delta 3200", "Kodak T-Max 100",
    "Lomography Color Negative 800", "Kodak Ektachrome E100",
    "Fujifilm Provia 100F", "Kodak Aerochrome infrared",
    "expired Polaroid 600", "Kodak Plus-X 125", "Agfa Vista 200",
    "digital RAW — no film emulation", "ACES log with cinema grade",
]

LIGHTING = [
    "natural light only", "single key light with hard shadows",
    "soft diffused window light", "neon practical lights",
    "chiaroscuro — dramatic light and shadow", "backlit with lens flare",
    "overhead fluorescent", "candlelight", "golden hour rim light",
    "moonlight", "mixed color temperature practicals",
    "Rembrandt lighting", "butterfly lighting", "split lighting",
    "silhouette against bright background", "volumetric god rays",
    "LED panel — cool daylight", "tungsten warm key",
    "cross-lit with colored gels", "dappled light through trees",
    "stadium floodlights", "campfire glow", "lightning flash",
    "car headlights cutting through fog", "string lights and fairy lights",
    "stained glass colored light", "fireplace amber glow",
    "reflected light off water", "bioluminescence", "neon sign spill",
]

SUBJECT = [
    "a lone figure in silhouette", "two people in conversation",
    "an elderly person's weathered hands", "a dancer mid-leap",
    "a child discovering something", "a musician performing",
    "a chef at work in a kitchen", "a person reading by a window",
    "an athlete in motion", "a couple walking away from camera",
    "a street vendor at their stall", "a dog waiting patiently",
    "a reflection in a puddle", "an empty chair", "hands holding coffee",
    "a woman looking out a train window", "a fisherman casting a line",
    "a painter at an easel", "a mechanic under a car hood",
    "someone running in the rain", "a person meditating",
    "a bartender mixing a drink", "a crowd from above",
    "a skateboarder mid-trick", "flowers in a cracked vase",
    "a vintage car on an empty road", "a surfer carrying a board",
    "a person standing at a crossroads", "an old man playing chess alone",
    "a nurse at the end of a long shift", "a lighthouse keeper",
    "a busker playing guitar", "typewriter with a half-written letter",
    "a tattooed hand holding a cigarette", "shoes on a telephone wire",
    "a ballerina tying her pointe shoes", "a welder behind a mask of sparks",
    "an astronaut looking back at Earth", "a grandmother cooking",
    "someone walking through falling leaves",
]

COLOR_PALETTE = [
    "warm earth tones — amber, ochre, burnt sienna",
    "cool blue and teal monochrome",
    "high contrast black and white",
    "desaturated with single color accent — red",
    "neon cyberpunk — magenta, cyan, electric blue",
    "pastel dreamscape — lavender, blush, mint",
    "autumn palette — rust, gold, deep green",
    "bleach bypass — muted with crushed blacks",
    "Technicolor vintage — oversaturated primaries",
    "Nordic minimal — pale blue, white, gray",
    "golden and bronze tones",
    "midnight palette — deep navy, purple, black",
    "sunset gradient — peach, coral, violet",
    "forest palette — emerald, moss, bark brown",
    "industrial — gunmetal, concrete, rust",
    "tropical — turquoise, coral, palm green",
    "sepia and cream",
    "cross-processed — shifted greens and magentas",
    "duotone — two color only",
    "natural ungraded — true to life colors",
]

COMPOSITION = [
    "rule of thirds", "centered symmetry", "leading lines",
    "Dutch angle", "bird's eye view", "worm's eye view",
    "over-the-shoulder", "extreme close-up", "wide establishing shot",
    "medium close-up", "cowboy shot", "full body with negative space",
    "frame within a frame", "shallow depth of field — bokeh background",
    "deep focus — everything sharp", "foreground framing element",
    "reflection composition", "diagonal tension", "golden ratio spiral",
    "minimalist with vast negative space", "layered depth — fore/mid/background",
    "point of view shot", "low angle hero shot", "high angle vulnerability",
    "split screen composition", "rack focus implied",
    "Wes Anderson centered symmetry", "Kubrick one-point perspective",
]

ERA_STYLE = [
    "1920s Art Deco", "1940s film noir", "1950s Technicolor",
    "1960s French New Wave", "1970s gritty realism", "1980s neon excess",
    "1990s grunge", "2000s digital clean", "2010s Instagram aesthetic",
    "contemporary editorial", "timeless and era-ambiguous",
    "retro-futurism", "Y2K aesthetic", "vaporwave",
    "cottagecore", "dark academia", "Afrofuturism",
    "solarpunk", "brutalist", "maximalist baroque",
]


class DigitRandomPrompt:
    """Generates randomized cinematic image prompts from mix-and-match categories."""

    CATEGORY_MAP = {
        "setting": SETTING,
        "location": LOCATIONS,
        "time_of_day": TIME_OF_DAY,
        "weather": WEATHER,
        "mood": MOOD,
        "camera": CAMERA,
        "lens": LENS,
        "film_stock": FILM_STOCK,
        "lighting": LIGHTING,
        "subject": SUBJECT,
        "color_palette": COLOR_PALETTE,
        "composition": COMPOSITION,
        "era_style": ERA_STYLE,
    }

    CATEGORY = "DIGIT"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"

    @classmethod
    def INPUT_TYPES(cls):
        all_option = ["random"]

        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "setting": (all_option + SETTING, {"default": "random"}),
                "location": (all_option + LOCATIONS, {"default": "random"}),
                "time_of_day": (all_option + TIME_OF_DAY, {"default": "random"}),
                "weather": (all_option + WEATHER, {"default": "random"}),
                "mood": (all_option + MOOD, {"default": "random"}),
                "subject": (all_option + SUBJECT, {"default": "random"}),
                "camera": (all_option + CAMERA, {"default": "random"}),
                "lens": (all_option + LENS, {"default": "random"}),
                "film_stock": (all_option + FILM_STOCK, {"default": "random"}),
                "lighting": (all_option + LIGHTING, {"default": "random"}),
                "color_palette": (all_option + COLOR_PALETTE, {"default": "random"}),
                "composition": (all_option + COMPOSITION, {"default": "random"}),
                "era_style": (all_option + ERA_STYLE, {"default": "random"}),
                "custom_prefix": ("STRING", {"default": "", "multiline": True, "tooltip": "Text prepended to the generated prompt."}),
                "custom_suffix": ("STRING", {"default": "", "multiline": True, "tooltip": "Text appended to the generated prompt."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed = kwargs.get("seed", 0)
        if seed == 0:
            return float("nan")
        return seed

    def generate_prompt(
        self,
        seed,
        setting="random",
        location="random",
        time_of_day="random",
        weather="random",
        mood="random",
        subject="random",
        camera="random",
        lens="random",
        film_stock="random",
        lighting="random",
        color_palette="random",
        composition="random",
        era_style="random",
        custom_prefix="",
        custom_suffix="",
    ):
        rng = random.Random(seed if seed > 0 else random.randint(1, 2147483647))

        def pick(value, options):
            if value == "random":
                return rng.choice(options)
            return value

        s_setting = pick(setting, SETTING)
        s_location = pick(location, LOCATIONS)
        s_time = pick(time_of_day, TIME_OF_DAY)
        s_weather = pick(weather, WEATHER)
        s_mood = pick(mood, MOOD)
        s_subject = pick(subject, SUBJECT)
        s_camera = pick(camera, CAMERA)
        s_lens = pick(lens, LENS)
        s_film = pick(film_stock, FILM_STOCK)
        s_lighting = pick(lighting, LIGHTING)
        s_palette = pick(color_palette, COLOR_PALETTE)
        s_comp = pick(composition, COMPOSITION)
        s_era = pick(era_style, ERA_STYLE)

        parts = [
            f"{s_subject}",
            f"{s_setting}, {s_location}",
            f"{s_time}, {s_weather}",
            f"{s_mood} mood, {s_era} style",
            f"{s_lighting}",
            f"{s_comp} composition",
            f"shot on {s_camera} with {s_lens}",
            f"{s_film}",
            f"color palette: {s_palette}",
        ]

        prompt = ". ".join(parts)

        if custom_prefix and custom_prefix.strip():
            prompt = f"{custom_prefix.strip()}. {prompt}"
        if custom_suffix and custom_suffix.strip():
            prompt = f"{prompt}. {custom_suffix.strip()}"

        return (prompt,)

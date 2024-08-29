import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def categ(x):
    country = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",
    "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
    "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad",
    "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Cote d'Ivoire", "Croatia",
    "Cuba", "Cyprus", "Czech", "Democratic Republic of the Congo", "Denmark", "Djibouti",
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hong Kong", "Hungary",
    "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho",
    "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
    "Mali", "Malta", "Marshall Islands", "Martinique", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
    "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia",
    "Norway", "Oman", "Pakistan", "Palau", "Palestine State", "Panama", "Papua New Guinea", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia",
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom","UK", "United States of America","USA","United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
    ] 

    # Mapping the categories to broader categories
    mapping = {
        'Breaking News': [
            'COVID', 'News', 'Breaking News', 'nan'  # General breaking news items
        ],
        'Politics': [
            'Politics', 'Palestine, State of', 'United States', 'Taiwan, Province of China', 
            'Uruguay', 'Virgin Islands, U.S.', 'Viet Nam', 'Korea, Republic of',
            'Russian Federation', 'Iran, Islamic Republic of'
        ],
        'World News': [
            'world', 'country', 'Congo, The Democratic Republic of the', 'America', 
            'Africa', 'Europe', 'Asia', 'Guam', 'Aruba', 'Guernsey', 'Antarctica',
            'Bermuda', "Côte d'Ivoire", 'Christmas Island', 'Cayman Islands', 
            'Gibraltar', 'Greenland', 'Isle of Man', 'Jersey', 'Macao', 'Réunion', 
            'Montserrat', 'Puerto Rico', 'Martinique'
        ],
        'Business/Finance': [
            'Business/Finance', 'Real estate', 'Stock', 'Startups', 'Finance', 'Entrepreneurship',
            'Cryptocurrency', 'Blockchain', 'Bitcoin', 'Amazon', 'Jobs'
        ],
        'Technology': [
            'Technology', 'Artificial Intelligence', 'Google', 'YouTube', 'Facebook', 
            'TikTok', 'Instagram', 'Virtual Reality', 'Coding', 'Design'
        ],
        'Science': [
            'Science', 'Space', 'Astronomy'
        ],
        'Food' : ['Food'],
        'Health': [
            'Health', 'Fitness', 'Nutrition', 'Vegan', 'Mindfulness', 'Meditation',
            'Yoga', 'Happiness', 'Philosophy', 'Psychology', 'Parenting', 'Motivation'
        ],
        'Entertainment': [
            'Entertainment', 'Movies', 'Music', 'Anime', 'Podcasts', 'Poetry', 
            'Games', 'Art', 'Photography', 'Beauty', 'Fashion', 'Love'
        ],
        'Sports': [
            'Sports', 'Hiking', 'Cars'
        ],
        'Environment': [
            'Environment', 'Climate', 'Gardening', 'Sustainability'
        ],
        'Crime': [
            # Since no explicit categories fit into 'Crime', we leave it empty for now.
        ],
        'Education': [
            'Education', 'History', 'Home', 'Recipes', 'DIY'
        ],
        'Weather': [
            'Weather'
        ],
        'Other': [
            'Other', 'Architecture', 'Minimalism', 'Productivity', 'Travel', 'Mindfulness', 
            'Christmas Island', 'Isle of Man', 'Guam', 'Aruba', 'Guernsey', 'Antarctica',
            'Bermuda', 'Côte d\'Ivoire', 'Christmas Island', 'Cayman Islands', 
            'Gibraltar', 'Greenland', 'Jersey', 'Macao', 'Réunion', 'Relationships', 'Pets'
        ]
    }

    x = str(x)
    cntry = [i.lower() for i in country]
    if x.lower() in cntry:
        x =  'country'

    for i in mapping.keys():
        if x in mapping[i]:
            return i
    return f'none yet - {x}'

a = r"adaasfasasfasdasdasdasdasdasddsfdsfssadsadasdsadasdasdadaasfasasfasdasdasdasdasdasddsfdsfsdfdfsf\nasdasddfdfsf\nasdasd[BEGIN]1. The Industrial Revolution and its consequences have been a disaster for the human race. They have greatly increased the life-expectancy of those of us who live in “advanced” countries, but they have destabilized society, have made life unfulfilling, have subjected human beings to indignities, have led to widespread psychological suffering (in the Third World to physical suffering as well) and have inflicted severe damage on the natural world. The continued development of technology will worsen the situation. It will certainly subject human beings to greater indignities and inflict greater damage on the natural world, it will probably lead to greater social disruption and psychological suffering, and it may lead to increased physical suffering even in “advanced” countries.[END]sadsadasdasjkgdkjashdashjkdka"
try:
    #get the substring of a starting with the string [begin] and ending with string [end]
    a = a[a.index("[BEGIN]")+7:a.index("[END]")]
except:
    print("error")
print(a)
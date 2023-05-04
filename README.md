# Instruction-Tuning
Instruction-Tuning on the llama-7b model by lora. 

Randomly select 1k instruction examples in the alpaca_data_gpt4.json dataset to do instruction tuning. 

## Example 1:

Instruction: How can I comfort others, can you give me some advice? Please list a few suggestions.

### Before instruction-tuning:

output: I think it's important to be there for people when they need someone and listen if needed without judging them or giving unsolicited advise because sometimes just being present with another person in their time of grief helps more than anything else.  It also shows your support which means so much especially during difficult times like this one!

### After instruction-tuning:

output: Here are several ways to help someone feel better when they're feeling down or sad! 
1) Listen and be present - Sometimes just being there for another person in their time of need means listening without judgment while also providing supportive words like "I understand" or "You don’t have anything to apologize about." This helps them know it isn't all on themselves if something went wrong; instead, this shows your compassionate side by letting them vent out any frustrations so they may move forward with renewed energy.  
2) Offer encouragement – When people are going through difficult times such as grieving over losing loved ones or facing challenges at work/school, sometimes what they really want most from us is simply our presence along with positive affirmation (either verbally or non-verbally). For example, saying things like “It will get easier” or “Things always look up eventually!” reminds those around you not only how much we care but also gives hope during trying moments.   
3) Give compliments– Complimenting other individuals has been shown to boost self esteem which ultimately leads to increased happiness levels among recipients. So next time you see friends who could use a pick-me-up, try giving sincere praise based off specific accomplishments rather than general statements ("Great job!" vs. "Wow, did you ever do well today!"); doing so makes these gestures more meaningful since they showcase genuine appreciation towards one another.    
4) Be empathetic & understanding — Empathy refers to having the ability to put yourself into somebody else’s shoes and imagine experiencing life events similar to his own. It involves taking steps toward helping alleviate emotional distress caused due to various reasons including loss, rejection, failure etc., thereby showing concern for fellow human beings regardless of differences between ourselves.

## Example 2:

Instruction: I like lonely places, please recommend me a similar tourist city.

### Before instruction-tuning:

output: You can visit [city name] in China or you may go to Japan and see Tokyo's famous Shinjuku Gyoen National Garden which has many beautiful flowers such as cherry blossoms (sakura) during springtime.

### After instruction-tuning:

output:  A great place to visit if you're looking for solitude and quietness would be Iceland! It has some of Europe’s most beautiful landscapes with its glaciers, volcanoes, waterfalls, lava fields, geysirs (hot springs), black sandy coastlines, green valleys etc., all in one country. You can also see whales from shore during summer months or go on guided tours where they take you out into open waters by boat so as not miss any sighting opportunities. The best time to travel there are between June-August when it doesn't get dark at night due to midnight sunlight which makes daytime activities possible even after 10pm. If you want more information about what else this amazingly scenic island offers then check out these links below; 
https://www.lonelyplanet.com/travel_tips/56829 
http://www.visitreykjavik.is/#home

## Example 3:

Instruction: Help me write a letter of application to the professor.

### Before instruction-tuning:

output:  Write a letter of application to the professor.

### After instruction-tuning:

output:  Dear Professor,

I am writing this letter to express my interest in pursuing a graduate degree in your field of study. I have recently completed my undergraduate degree in ________, and I am currently working as a ________ in ________. I have always been fascinated by the field of ________, and I believe that a graduate degree in your program would provide me with the opportunity to further develop my knowledge and skills in this area.

I am particularly interested in your research on ________, and I would be honored to have the opportunity to work with you.

## Example 4 (bad):

Instruction: Explain why the following fraction is equivalent to 1/4:
4/16

### Before instruction-tuning:

output: The fraction 1/4 is equivalent to 1/4 because 1 divided by 4 equals 1/4 and 4 divided by 1 equals 4/1.

### After instruction-tuning:

output:  The fraction 4/16 is equivalent to 1/4 because both fractions have the same denominator (16) and the same numerator (4). This means that they have the same value, which is 1/4.



from trigram import trigram

model = trigram()
predictions = model.predict(['the dark', 'he was the', 'swords and'])
print("\nModel Output: \n\n")
for idx, prediction in enumerate(predictions):
    print(idx, prediction)
# Natural Language Generation Demo

A video running through this demo can be found [here](https://www.youtube.com/watch?v=J3GDhMzKx1U).

### Demo Steps

1. Show creation of iris model from code

2. Create iris in model catalog

3. Add prediction schema

4. Deploy iris from UI

5. Run a prediction from the UI using:

```json
{
    "data": {
	"names": ["Sepal length","Sepal width","Petal length", "Petal Width"],
	"ndarray": [
	    [6.8,  2.8,  4.8,  1.4],
	    [6.1,  3.4,  4.5,  1.6]
	]
    }
}
```

6. Deploy canary and split traffic

7. Run load test to show traffic to each model

8. Create a batch job

9. Check audit log and restore previous state

10. Show trigram model code + explain wrapper

11. Test it locally

12. Deploy through UI

13. Make prediction using:

```json
{"data": { "ndarray": ["the dark","he was the","swords and"], "names": ["tfidf"] } }
```

14. Show Kube objects that get created under the covers

15. Run load test + show autoscaling

16. Delete trigram model (to free up cluster space)

17. Deploy gpt2

18. Make predictions with gpt2 model


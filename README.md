#Machine Learning
Linear Regression and Logistic Regression implemented by me. 

Celem zadania jest implementacja regresji liniowej oraz logistycznej. Wagi powinny być liczone za pomocą metody gradient descent.
Implementacja ma dostarczać dwóch klas: MyLogisticRegression oraz MyLinearRegression, tzw. estymatorów w scikit-learn (http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator). Tzn. z zaimplementowanych przez nas estymatorów musi się dać korzystać zgodnie z interfejsem estymatorów z scikit-learn (fit, predict).
Ponadto, trzeba obsłużyć szereg parametrów:
* `batch_size (int)` ilość przypadków treningowych w jednym batchu (cały zbiór jeżeli nie podano) [gradient liczymy na jednym batchu i na jego podstawie robimy update wag]
* `n_epochs (int)` ilośc epok (jedna epoka = przejrzenie wszystkich przypadków treningowych, dokładnie raz)
* `shuffle (bool)` w przypadku True, w każdej iteracji kolejność przypadków powinna być losowa
* `holdout_size (float)` frakcja tej wielkości zostanie odłożona jako zbiór walidacyjny
* `l2 (float)` współczynnik regularyzacji l2
* `learning_rate (float)` learning rate
* `decay (float)` decay dla learning rate (tzn. co epokę przemnażamy learning_rate przez decay)
* `standardize` standaryzacja danych przed uczeniem (średnia 0 i wariancja 1)  

Uwaga:
domyślne wartości parametrów powinny być sensowne (dawać sensowne wyniki na standardowych problemach),
należy zaimplementować (i dostarczyć) również moduł test.py, w którym testujemy naszą implementację.

import pandas as pd
from models.OneR import OneR
from models.ZeroR import ZeroR


path = input("Ingrese la ruta del archivo: ")
dataset = pd.read_csv(path)

print("Dataset:")
print(dataset)

print("Ingrese el nombre de la clase objetivo: ")
target = input()
print("Clase:",target)

iterations=int(input("Ingrese el número de iteraciones: "))

print("--------------------ZeroR-------------------")

zero_r = ZeroR(dataset, target)
zero_r.fit()



most_common_class = zero_r.most_common_class
print(f"La clase más común es: {most_common_class}")


avg_accuracy = zero_r.evaluate(n_iterations=iterations)
print(f"Precisión promedio: {avg_accuracy:.2f}")



print("--------------------OneR----------------------")


one_r = OneR(dataset, target)
one_r.fit()

avg_accuracy = one_r.evaluate(n_iterations=iterations)
print(f"Precisión promedio: {avg_accuracy:.2f}")
print(f"La mejor característica es: {one_r.best_feature}")
print("Reglas:")
for value, most_common_class in one_r.best_rules.items():
    print(f"Si {one_r.best_feature} es {value}, entonces la clase es {most_common_class}")
from system_clasificacion import *
import unittest 

class test(unittest.TestCase):
    def Test_clasificacion(self):
        sistema = Modelo_clasificacion()
        Resultado = sistema.modeloClasificacion()
        self.assertTrue(Resultado["Resultado"],"Modelo Ejecutado correctamente")
        self.assertGreaterEqual(Resultado["Precision"],0.7,"La precisión del modelo debe ser mayor a 0.7")

if __name__ == "__main__":
    unittest.main()

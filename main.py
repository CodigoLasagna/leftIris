import cv2
import numpy as np

def contar_objetos(image_path):
    try:
        # Cargar la imagen
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError("No se pudo cargar la imagen")

        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbralización adaptativa a la imagen en escala de grises
        adaptive_threshold = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 4)
        
        # Encontrar los contornos en la imagen binarizada
        contours, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Inicializar lista para almacenar posiciones de objetos
        objetos = []
        
        # Encontrar los rectángulos delimitadores para los contornos
        for contour in contours:
            # Calcular el rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Descartar objetos muy pequeños y el fondo
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            # Guardar la posición del objeto
            objetos.append((x, y, w, h))
        
        return image, objetos

    except Exception as e:
        print(f"Error: {e}")
        return None, None

def contar_colores(image, objetos):
    # Contador de objetos por color
    count = {}
    
    # Contar objetos por color
    for x, y, w, h in objetos:
        # Obtener la región de interés (ROI)
        roi = image[y:y+h, x:x+w]

        # Calcular el color promedio para la región de interés
        avg_color = cv2.mean(roi)[:3]
        avg_color = tuple(map(int, avg_color))
        
        # Guardar la posición y el color promedio del objeto
        count[(x, y, w, h)] = avg_color
    
    return count

def agrupar_colores(colors, threshold=40):
    # Agrupar colores similares
    grouped_colors = {}
    for coords, color in colors.items():
        # Buscar un grupo existente para el color
        found_group = False
        for group_color, group_coords in grouped_colors.items():
            if np.all(np.abs(np.array(color) - np.array(group_color)) < threshold):
                grouped_colors[group_color].append(coords)
                found_group = True
                break
        # Si no se encontró un grupo, crear uno nuevo
        if not found_group:
            grouped_colors[color] = [coords]
    
    return grouped_colors

# Función para imprimir el conteo de objetos por color
def imprimir_conteo(objetos_por_color):
    print("Cantidad de objetos por color:")
    for color, objetos in objetos_por_color.items():
        print(f"Color {color}: {len(objetos)}")
    total = sum(len(objetos) for objetos in objetos_por_color.values())
    print(f"Cantidad total: {total}")

def dibujar_rectangulos(image, objetos, colors):
    counter = 1
    for (x, y, w, h), avg_color in colors.items():
        # Dibujar rectángulo alrededor de cada objeto con su respectivo color
        cv2.rectangle(image, (x, y), (x + w, y + h), avg_color, 2)
        
        # Escribir el número del objeto
        cv2.putText(image, str(counter), (x+8, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (17, 17, 17), 5)
        cv2.putText(image, str(counter), (x+8, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        counter += 1
    
    return image

# Ruta de la imagen
image_path = "dotsCol.jpg"

# Contar objetos y obtener posiciones
image, objetos = contar_objetos(image_path)

if image is not None and objetos is not None:
    # Contar objetos por color
    obj_colors = contar_colores(image, objetos)

    # Agrupar colores similares
    grouped_colors = agrupar_colores(obj_colors)

    # Imprimir conteo de objetos por color
    imprimir_conteo(grouped_colors)

    # Dibujar rectángulos alrededor de los objetos
    image_with_rectangles = dibujar_rectangulos(image.copy(), objetos, obj_colors)

    # Mostrar imagen de salida
    cv2.imshow("Output", image_with_rectangles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# membrane curvature calculation
# The influence of hydroxylation of sphingolipids on membrane physical state
# Lucia Sessa, Stefano Piotto, Francesco Marrafino, Barbara Panunzi, Rosita Diana, Simona Concilio
# lucsessa@unisa.it

# @title 1. import libraries and read files
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

file_path = '/file/path.txt'

data = np.loadtxt(file_path, skiprows=2)
x_values = data[:, 0]
y_values = data[:, 1]
z_values = data[:, 2]

points = np.column_stack((x_values, y_values, z_values))
min_distance = 0.1
filtered_indices = []
filtered_points = []

for i in range(len(points)):
    too_close = False
    for j in range(len(filtered_points)):
        distance = np.linalg.norm(points[i] - filtered_points[j])
        if distance < min_distance:
            too_close = True
            break
    if not too_close:
        filtered_indices.append(i)
        filtered_points.append(points[i])

filtered_points = np.array(filtered_points)
filtered_indices = np.array(filtered_indices)

# Calculates the minimum and maximum values for each dimension
min_value_x, max_value_x = np.min(filtered_points[:, 0]), np.max(filtered_points[:, 0])
min_value_y, max_value_y = np.min(filtered_points[:, 1]), np.max(filtered_points[:, 1])
min_value_z, max_value_z = np.min(filtered_points[:, 2]), np.max(filtered_points[:, 2])

# @title 2. Triangulation
#!pip install plotly
from scipy.spatial import Delaunay
import plotly.graph_objects as go

# Perform Delaunay triangulation
tri = Delaunay(filtered_points[:, [0, 2]])  # Considera solo le coordinate x e z

# Extract nodes
nodes = filtered_points

# Create a Scatter3d plot for nodes
scatter = go.Scatter3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    mode='markers',
    marker=dict(size=1, color='red')
)

# Create a Mesh3d plot for the Convex Hull surface
mesh = go.Mesh3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    i=tri.simplices[:, 0],
    j=tri.simplices[:, 1],
    k=tri.simplices[:, 2],
    opacity=0.1,
    color='cyan'
)

# Create a Scatter3d plot for the sides of triangles
edge_scatter = go.Scatter3d(
    x=np.concatenate([nodes[simplex, 0] for simplex in tri.simplices]),
    y=np.concatenate([nodes[simplex, 1] for simplex in tri.simplices]),
    z=np.concatenate([nodes[simplex, 2] for simplex in tri.simplices]),
    mode='lines',
    line=dict(color='black', width=2)
)

# Creating a layout
layout = go.Layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')
    )
)

# Calculates the minimum and maximum values for each dimension
min_value_x, max_value_x = np.min(nodes[:, 0]), np.max(nodes[:, 0])
min_value_y, max_value_y = np.min(nodes[:, 1]), np.max(nodes[:, 1])
min_value_z, max_value_z = np.min(nodes[:, 2]), np.max(nodes[:, 2])

# Update the layout with the calculated values
layout.scene.xaxis.range = [min_value_x, max_value_x]
layout.scene.yaxis.range = [-20, 50]
layout.scene.zaxis.range = [min_value_z, max_value_z]

# Create a figure
fig = go.Figure(data=[scatter, mesh, edge_scatter], layout=layout)

# @title 3.  Gaussian curvature calculation
def Gaussian(tri, points):
    gaussian_values = []

    for i in range(len(points)):
        vertex_triangles = [tri.simplices[j] for j in range(len(tri.simplices)) if i in tri.simplices[j]]

        sum_angles = 0.0
        sum_areas = 0.0

        for triangle in vertex_triangles:
            # Find the indices of the current point in the triangle.
            vertex_index = np.where(triangle == i)[0][0]

            # Get the indices of the other two vertices in the triangle.
            other_vertices = np.delete(triangle, vertex_index)

            # Calculate the angles at the vertices of the triangle.
            v1 = points[other_vertices[0]] - points[i]
            v2 = points[other_vertices[1]] - points[i]
            dot_product = np.dot(v1, v2)
            cross_product = np.linalg.norm(np.cross(v1, v2))
            angle = np.arctan2(cross_product, dot_product)

            sum_angles += angle

            # Calculate the area of the triangle.
            area = 0.5 * cross_product
            sum_areas += area

        gaussian_value = (2 * np.pi - sum_angles) / sum_areas /3 if sum_areas != 0 else 0.0
        gaussian_values.append(gaussian_value)

    return np.array(gaussian_values)

gaussian_values = Gaussian(tri, points)


file_name = file_path.split("/")[-1].split(".")[0]  # Extracting file name without extension
output_file_name = f"{file_name}_Gaussian.txt"
output_file_path = f"/content/{output_file_name}"  # Replace this with the desired output path
np.savetxt(output_file_path, gaussian_values, delimiter=',', header='Gaussian Curvature Values', comments='')

#print(f"Gaussian curvature values saved to {output_file_path}")

# @title 4. Mean curvature calculation
import numpy as np
from scipy.spatial import Delaunay
import plotly.graph_objects as go

# Perform Delaunay triangulation
tri = Delaunay(filtered_points[:, [0, 2]])  # Consider only x and z coordinates.

# Define the function to calculate the curvature H
def calculate_H_curvature(tri, points):
    H_curv_values = []
    # Get the vertex indices of triangles from triangulation.
    simplices = tri.simplices
    for simplex in simplices:
        # Initialize the list of edges for each new triangle.
        edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]

        for edge in edges:
            # Extract edge vertex indices.
            vertex1, vertex2 = edge

            # Find triangles adjacent to the edge.
            triangles = []
            for other_simplex in simplices:
                if vertex1 in other_simplex and vertex2 in other_simplex:
                    triangles.append(other_simplex)

            # Calculate the angle between the two adjacent triangles (theta).
            v1 = points[vertex1] - points[vertex2]
            v1 /= np.linalg.norm(v1)
            angles = []
            for triangle in triangles:
                other_vertex = [v for v in triangle if v != vertex1 and v != vertex2][0]
                v2 = points[other_vertex] - points[vertex2]
                v2 /= np.linalg.norm(v2)
                dot_product = np.dot(v1, v2)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                angles.append(angle)
            theta = np.sum(angles)

            # Calculate the length of the edge (Length_edge).
            Length_edge = np.linalg.norm(points[vertex1] - points[vertex2])

            # Calculate the areas of adjacent triangles.
            area_tri_adj = 0.0
            for triangle in triangles:
                v0 = points[triangle[0]]
                v1 = points[triangle[1]]
                v2 = points[triangle[2]]
                cross_product = np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                area_tri_adj += cross_product / 2.0

            # Calculate H_curv
            H_curv = theta * Length_edge / (2 * area_tri_adj)
            H_curv_values.append(H_curv)

    return np.array(H_curv_values)

# Calculates the curvature values H
H_curv_values = calculate_H_curvature(tri, filtered_points)

# Write index and curvature values to a text file
output_file_name_indices = f"{file_name}_H_curvature.txt"
output_file_path_indices = f"/content/{output_file_name_indices}"  # Replace this with the desired output path

with open(output_file_path_indices, 'w') as file:
    file.write("Index, H_curvature\n")
    for i, curvature in enumerate(H_curv_values):
        file.write(f"{i + 1}, {curvature}\n")

# @title 5. plots
# Calculate mean and standard deviation of H_curv_values and Gaussian_values
mean_H_curv = np.mean(H_curv_values)
std_dev_H_curv = np.std(H_curv_values)

mean_gaussian = np.mean(gaussian_values)
std_dev_gaussian = np.std(gaussian_values)

print("Mean H_curvature:", mean_H_curv)
print("Standard Deviation H_curvature:", std_dev_H_curv)

print("Mean Gaussian Curvature:", mean_gaussian)
print("Standard Deviation Gaussian Curvature:", std_dev_gaussian)

# Plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.hist(H_curv_values, bins=20, color='blue', edgecolor='black')
ax1.set_title('Distribution of H_curvature Values')
ax1.set_xlabel('H_curvature')
ax1.set_ylabel('Frequency')

ax2.hist(gaussian_values, bins=20, color='red', edgecolor='black')
ax2.set_title('Distribution of Gaussian Curvature Values')
ax2.set_xlabel('Gaussian Curvature')
ax2.set_ylabel('Frequency')

plt.show()

# @title 6. 3D chart part 1.
# Create a Mesh3d plot with color gradient based on gaussian_values.

# Multiply curvature values for display reasons.
gaussian_values_scaled = gaussian_values * 5000

mesh_with_color = go.Mesh3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    i=tri.simplices[:, 0],
    j=tri.simplices[:, 1],
    k=tri.simplices[:, 2],
    intensity=gaussian_values_scaled,  # Use gaussian_values for color gradient.
    opacity=0.7,
    colorscale='Rainbow',  
    colorbar=dict(title='Gaussian Values', tickvals=np.linspace(0.00, 20, 5)),
    cmin=0.00,
    cmax=10 
)

fig_with_color = go.Figure(data=[mesh_with_color], layout=layout)

# Show graph with color gradient
fig_with_color.show()

# @title 7. 3D chart part 2.
from plotly.subplots import make_subplots

# Calculate Gaussian curvature values (assuming 'tri' and 'nodes' are already defined)
gaussian_values = Gaussian(tri, nodes)

gaussian_values_scaled = gaussian_values * 5000
H_curv_values_scaled = H_curv_values * 5000

#######################
# Define the camera parameters
camera = dict(
    eye=dict(x=1.5, y=1.5, z=1.5),  # Eye position in 3D space
    center=dict(x=0, y=0, z=0),      # Center of the scene
    up=dict(x=0, y=1, z=0)           # Up direction
)

# Update the layout with the camera parameters
layout.scene.camera = camera
#######################

# Creating the first plot
mesh_with_color_1 = go.Mesh3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    i=tri.simplices[:, 0],
    j=tri.simplices[:, 1],
    k=tri.simplices[:, 2],
    intensity=gaussian_values_scaled, 
    opacity=0.7,
    colorscale='Rainbow', 
    colorbar=dict(title='Gaussian Values', tickvals=np.linspace(0.00, 20, 5)),
    cmin=0.00,
    cmax=10   
)

# Creating the second plot 
mesh_with_color_2 = go.Mesh3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=nodes[:, 2],
    i=tri.simplices[:, 0],
    j=tri.simplices[:, 1],
    k=tri.simplices[:, 2],
    intensity=H_curv_values_scaled,
    opacity=0.7,
    colorscale='Rainbow', 
    colorbar=dict(title='Mean Values', tickvals=np.linspace(0.00, 20, 5)),
    cmin=0.00,
    cmax=10
)

fig_with_color_1 = go.Figure(data=[mesh_with_color_1], layout=layout)
fig_with_color_2 = go.Figure(data=[mesh_with_color_2], layout=layout)

fig_with_color_1.show()
fig_with_color_2.show()

# @title 8. save graphs in html.
output_file_name_html_1 = f"{file_name}_Gaussian.html"
output_file_path_html_1 = f"/content/{output_file_name_html_1}"  # Replace this with the desired output path
go.Figure.write_html(fig_with_color_1,output_file_path_html_1) # write as html or image

output_file_name_html_2 = f"{file_name}_H_val.html"
output_file_path_html_2 = f"/content/{output_file_name_html_2}"  # Replace this with the desired output path
go.Figure.write_html(fig_with_color_2,output_file_path_html_2) # write as html or image
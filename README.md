# Point_Cloud_Deep_Learning
Deep Learning on Point Clouds

Point clouds inherently lack topological information and thus designing a model to recover topology can enrich the representation power of point clouds.



CNN are applied to Images(structured data)
DGCNN tries to achieve CNN based performance and features on pint clouds.
Here point clouds are first converted into Graphs and then convolutions are applied on these graphs.
As compared to pointnet where only *global geometric information* was only being utilised, DGCNN extends pointnet by also utilising the *local geometric information*.This is done by applying graph convolutions


DGCNN proposes EdgeConv module suitable for CNN based high level tasks on point clouds including classification and segmentation.
EdgeConv acts on graphs dynamically computed in each layer of the network.

EdgeConv has the following properties:
-It is differentiable
-It incorporates local neighborhood information
-It can be stacked to learn global shape properties
-In multi-layer systems affinity in feature space captures semantic characteristics over potentially long distances in original embedding

# Resources

https://www.youtube.com/watch?v=bBS1xLzqp1U
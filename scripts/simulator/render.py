## Import all relevant libraries
import sys
import bpy
import numpy as np
import math as m
import random
import os
from datetime import datetime
import fnmatch


## Main Class
class Render:
    def __init__(self):

        self._max_simulation_steps = 10
        self._nickname = 'paired_dataset'
        self._is_output_enabled = False

        # gio linux
        self._hdri_images_path = '/home/gionji/MDU/Blender/projects/syntetic_dataset_blender_generator_trafficsigns/textures/environment/'
        self._output_path = '/home/gionji/MDU/Datasets/blender_generated/'

        self._initial_camera_location = (0, -2, 0)

        self.obj_names = [ 'SL_10_KMH', 'SL_15_KMH', 'SL_20_KMH', 'SL_25_KMH', 'SL_30_KMH', 'SL_35_KMH', 
                           'SL_40_KMH', 'SL_45_KMH', 'SL_50_KMH', 'SL_55_KMH', 'SL_60_KMH', 'SL_65_KMH',
                           'SL_70_KMH', 'SL_75_KMH', 'SL_80_KMH', 'SL_90_KMH', 'SL_100_KMH', 'SL_110_KMH',
                           'SL_120_KMH', 'SL_130_KMH']

        ## Scene information
        # Define the scene information
        self.scene = bpy.data.scenes['Scene']

        # Define the information relevant to the <bpy.data.objects>
        self.camera = bpy.data.objects['Camera']
        self.axis = bpy.data.objects['Main Axis']
        self.light_1 = bpy.data.objects['Light1']
        self.light_2 = bpy.data.objects['Light2']
        
        self.projector_image_01 = bpy.data.objects['ImageProjector']
        self.projector_noise_01 = bpy.data.objects['NoiseProjector.001']
                         
        self.objects, self.objects_label = self.create_objects() # Create list of bpy.data.objects from bpy.data.objects[1] to bpy.data.objects[N]
  
        ## Output information
        # Input your own preferred location for the images and labels
        self.images_filepath = None
        self.labels_filepath = None
        
         # elouan windows
        #self.hdri_images_path = 'C:/Users/ecu01/Documents/Blender/Sign/textures/environment'
        #self.output_path = 'C:/Users/ecu01/Documents/Blender/output_sign/'

        # gio windows
        #self.hdri_images_path = 'C:/Users/gbi02/Desktop/syntetic_dataset_blender_generator_trafficsigns/textures/environment'
        #self.output_path = 'C:/Users/gbi02/Desktop/trafficsigns_yolo_datasets'

    
    # Setter for max_simulation_steps attribute
    @property
    def max_simulation_steps(self):
        return self._max_simulation_steps

    @max_simulation_steps.setter
    def max_simulation_steps(self, value):
        if value < 1:
            raise ValueError("max_simulation_steps must be a positive integer")
        self._max_simulation_steps = value

    # Setter for nickname attribute
    @property
    def nickname(self):
        return self._nickname

    @nickname.setter
    def nickname(self, value):
        if not isinstance(value, str):
            raise ValueError("nickname must be a string")
        self._nickname = value

    
    # Getter for the boolean attribute
    @property
    def is_output_enabled(self):
        return self._is_output_enabled

    # Setter for the boolean attribute
    @is_output_enabled.setter
    def is_output_enabled(self, value):
        if not isinstance(value, bool):
            raise ValueError("flag must be a boolean")
        self._is_output_enabled = value

    
    # Getter for hdri_images_path attribute
    @property
    def hdri_images_path(self):
        return self._hdri_images_path

    # Setter for hdri_images_path attribute
    @hdri_images_path.setter
    def hdri_images_path(self, value):
        if not isinstance(value, str):
            raise ValueError("hdri_images_path must be a string")
        self._hdri_images_path = value

    # Getter for output_path attribute
    @property
    def output_path(self):
        return self._output_path

    # Setter for output_path attribute
    @output_path.setter
    def output_path(self, value):
        if not isinstance(value, str):
            raise ValueError("output_path must be a string")
        self._output_path = value

    # Setter for nickname attribute
    @property
    def initial_camera_location(self):
        return self._initial_camera_location

    @nickname.setter
    def initial_camera_location(self, value):
        if not isinstance(value, tuple):
            raise ValueError("initial_camera_location must be a tuple")
        self._initial_camera_location = value



    ## Helpers methods
    def create_output_folder(self, path, nickname, 
                             images_subfolder_name='',
                             labels_subfolder_name=''):
        current_datetime = datetime.now()
        folder_name = current_datetime.strftime("%Y_%m_%d-%H_%M") + "-" + nickname
        folder_path = os.path.join(path, folder_name)
        
        self.images_filepath = os.path.join(folder_path , images_subfolder_name)
        self.labels_filepath = os.path.join(folder_path , labels_subfolder_name)        

        # Check if the folder doesn't exist
        if not os.path.exists(self.images_filepath):
            os.makedirs( self.images_filepath )

        # Check if the folder doesn't exist
        if not os.path.exists(self.labels_filepath):    
            os.makedirs( self.labels_filepath )            
        return
    
            
    def create_yolo_labels_file(self, classes_filename_subfolder='', classes_filename='classes'):
        file_path = os.path.join(self.images_filepath, classes_filename_subfolder, classes_filename+".txt")
        with open(file_path, "w") as file:
            for label in self.obj_names:
                file.write(label + "\n")
            file.flush()  # Flush the data to the file
            file.close()  # Close the file stream           
        return
            

    def pick_one_path(self, folder_path):
        # Get a list of all file names in the folder
        file_names = os.listdir(folder_path)
        # Randomly select a file name
        random_file_name = random.choice(file_names)
        # Create the full file path
        random_file_path = os.path.join(folder_path, random_file_name)

        print('background image:', random_file_name)

        return random_file_path        


    def list_materials_nodes_and_inputs( self ):
        #iterate on all SIGNS materials
        # Get all materials in the scene
        materials = bpy.data.materials

        # Iterate through each material
        for material in materials:
            # Check if the material has node-based shading
            if material.use_nodes:              
                print("\n\n--> Material Name: ", material.name)
                # Get the node tree of the material
                node_tree = material.node_tree
                # Get all nodes in the node tree
                nodes = node_tree.nodes
                # Iterate through each node
                for node in nodes:
                    # Check if the node is a material node
                    print("\n----> Node name: ", node.name, ' labeled: ', node.label)  
                    # Iterate on every input
                    for input in node.inputs:
                        print("------> Node input: ", input.name)                        
                        if input.name == 'Vector':
                            print("suca")
                        # Assume the node containing the parameter is named "ScaleNode"
                        #scale_node = bpy.data.node_groups["NodeGroup"].nodes["VectorNode"].inputs[0].links[0].from_node.inputs["ScaleX"].default_value = 2.0  

        return

    def delete_collection_objects(self, collection_name):
        # Get the collection by name
        collection = bpy.data.collections.get(collection_name)

        if collection is not None:
            # Loop through all objects in the collection and unlink them
            for obj in collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)

            # Unlink the collection itself
            bpy.data.collections.remove(collection, do_unlink=True)


    def delete_objects_by_name_pattern(self, pattern):
        objects_to_delete = [obj for obj in bpy.data.objects if fnmatch.fnmatch(obj.name, pattern)]
        for obj in objects_to_delete:
            bpy.data.objects.remove(obj, do_unlink=True)





## Scenes methods               
    #define the relative position of the camera from the empty axis
    def set_camera(self):
        # target axis position
        self.axis.rotation_euler = (0, 0, 0)
        self.axis.location = (0, 0, 0)
        # relative camera position from the "target" axis
        self.camera.location = self._initial_camera_location

    def dispose_objects(self, sign_distance=3.0):  
        # Get the collection containing the traffic signs
        blueprint_collection_name = "Traffic Signs"
        collection = bpy.data.collections.get(blueprint_collection_name)     
        self.delete_objects_by_name_pattern('SL_*_KMH.0*')
        if collection:
            # Create a new collection for the copied signs
            real_actors_scene_name = "Generated Scene"
            # Remove any previous copies of the collection if it exists
            existing_copied_collection = bpy.data.collections.get(real_actors_scene_name)           
            if existing_copied_collection:            
                try:
                    bpy.context.scene.collection.children.unlink(existing_copied_collection)
                finally:          
                    bpy.data.collections.remove(existing_copied_collection)         
            copied_collection = bpy.data.collections.new(real_actors_scene_name)
            bpy.context.scene.collection.children.link(copied_collection)
            
            original_objects = list(collection.objects.values())  # Convert dict values to a list
            random_permutation = random.sample(original_objects, len(original_objects))
            
            # Iterate over each sign in the original collection
            for i, obj in enumerate(random_permutation):
                # Copy the sign object
                copied_obj = obj.copy()
                copied_obj.data = obj.data.copy()
                # Link the copied object to the new collection
                copied_collection.objects.link(copied_obj)
                # Calculate the x and y position of the sign
                x = i * sign_distance - 15
                y = random.uniform(20, 20)
                z = random.uniform(0.0, 0.0)
                # Set the position of the copied sign
                copied_obj.location = (x, y, z)
            # Update the scene
            bpy.context.view_layer.update()
        self.objects, self.objects_label = self.create_objects() # Create list of bpy.data.objects from bpy.data.objects[1] to bpy.data.objects[N]       
        return

    
    def change_background_hdri_image(self, hdri_path, light_intensity=0.5) :
        # Get the current active scene
        scene = self.scene   

        # Check if the scene has a world
        if scene.world:
            # Get the world node tree
            world_node_tree = scene.world.node_tree
            
            ## Set the light emission density
            world_node_tree.nodes["Background"].inputs['Strength'].default_value = light_intensity
            ## Set the image source
            world_node_tree.nodes["Environment Texture"].image.filepath = self.pick_one_path( hdri_path )                    
        
        bpy.context.view_layer.update()
        return

    def edit_shadow_projector_parameters(self, 
            projector_obj, 
            position_range = ((-20, 20), (-40, 20), (2, 50)), 
            musgrave_scale_range = (2, 50),
            mapping_location_range = ((0, 10), (0, 10), (0, 10)),
            mapping_scale_range = ((0, 10), (0, 10), (0, 10)),
            color_ramp_range = ((0.0, 1.0), (0.0, 1.0)),
            emission_range = (9,10) ):
                
        proj_data = projector_obj.data
        proj_data.use_nodes = True 
        
        projector_obj.location = (random.randint(position_range[0][0], position_range[0][1]), 
            random.randint(position_range[1][0], position_range[1][1]), 
            random.randint(position_range[2][0], position_range[2][1]))

        #change the projection scale
        projector_obj.data.node_tree.nodes["Musgrave Texture"].inputs[2].default_value    = random.randint(musgrave_scale_range[0], musgrave_scale_range[1])
        #change the texture position
        projector_obj.data.node_tree.nodes["Mapping"].inputs['Location'].default_value    = (random.randint(mapping_location_range[0][0], mapping_location_range[0][1]),
             random.randint(mapping_location_range[1][0], mapping_location_range[1][1]),
             random.randint(mapping_location_range[2][0], mapping_location_range[2][1]))
        # change the texture scale
        projector_obj.data.node_tree.nodes["Mapping"].inputs['Scale'].default_value       = (random.randint(mapping_scale_range[0][0], mapping_scale_range[0][1]), 
            random.randint(mapping_scale_range[1][0], mapping_scale_range[1][1]),
            random.randint(mapping_scale_range[2][0], mapping_scale_range[2][1]))
        
        ## color ramp parameters
        #projector_obj.data.node_tree.node_tree.nodes["Color Ramp"].color_ramp.elements[0].position = random.uniform(color_ramp_range[0][0], color_ramp_range[0][1])
        #projector_obj.data.node_tree.node_tree.nodes["Color Ramp"].color_ramp.elements[1].position = random.uniform(color_ramp_range[1][0], color_ramp_range[1][1])
        
        # light strenght
        projector_obj.data.node_tree.nodes["Emission"].inputs[1].default_value = random.randint(emission_range[0], emission_range[1])

        bpy.context.view_layer.update()

        return
    


    def edit_snow_effect(self, gain=0):        
        #iterate on all SIGNS materials
        # Get all materials in the scene
        materials = bpy.data.materials

        # Iterate through each material
        for material in materials:
            # Check if the material has node-based shading and the material name contains on ofthe speed limits substring
            if material.use_nodes and any(s in material.name for s in self.obj_names) : 
                # Get the node tree of the material
                node_tree = material.node_tree
                # Get all nodes in the node treePrincipled
                nodes = node_tree.nodes
                # Select the active material. still confused but whatever works
                bpy.context.object.active_material.name = material.name
                  
                bpy.data.node_groups["Surface Effects Group"].nodes["ice_effect_ratio"].outputs[0].default_value =  1 - gain

                bpy.data.node_groups["Snow and Ice Group"].nodes["snow_ice_bsdf"].inputs[1].default_value = random.uniform(0.05, 0.20)                 
                material.node_tree.nodes["Principled BSDF"].inputs[1].default_value = 0       

                ## size of the snow
                bpy.data.node_groups["Snow and Ice Group"].nodes["Noise Texture"].inputs[2].default_value = random.uniform(0.1, 15) 

                ## Proportions of the snow texture
                bpy.data.node_groups["Snow and Ice Group"].nodes["Mapping.002"].inputs[3].default_value[0] = 0.8
                bpy.data.node_groups["Snow and Ice Group"].nodes["Mapping.002"].inputs[3].default_value[1] = 0.5

                # Color map. the depth of the snowflake
                #bpy.data.node_groups["Snow and Ice Group"].nodes["Snowflake depth"].color_ramp.elements[0].position = random.uniform(0.2, 0.6) 
                #bpy.data.node_groups["Snow and Ice Group"].nodes["Snowflake depth"].color_ramp.elements[1].position = random.uniform(0.2, 0.6) 

                ## effetto fico brina
                bpy.data.node_groups["Snow and Ice Group"].nodes["snow_ice_bsdf"].inputs[5].default_value = 0.736364

                ##  Update the scene
                bpy.context.view_layer.update()        

        return
    
    
    def edit_dirt_effect(self):
        bpy.data.node_groups["Surface Effects Group"].nodes["dirt_effect_ratio"].outputs[0].default_value = random.uniform(0.90, 1.0)    
        return
    


    def update_camera_position(self):
        self.camera.location = (0, -(5/10), 3)  
        bpy.context.view_layer.update()
        return


    def create_sample(self, render_counter, name_suffix=''):
        self.render_blender(render_counter, name_suffix)
        self.write_annotation_file(render_counter, name_suffix)
        return



    def write_annotation_file(self, render_counter, suffix=''):
        text_file_name = self.labels_filepath + '/' + str(render_counter) + '-' + suffix + '.txt' # Create label file name
        text_file = open(text_file_name, 'w+') # Open .txt file of the label     
        # Get formatted coordinates of the bounding boxes of all the objects in the scene
        # Display demo information - Label construction
        print("---> Label Construction")
        text_coordinates = self.get_all_coordinates()
        splitted_coordinates = text_coordinates.split('\n')[:-1] # Delete last '\n' in coordinates
        text_file.write('\n'.join(splitted_coordinates)) # Write the coordinates to the text file and output the render_counter.txt file
        text_file.close() # Close the .txt file corresponding to the label
 
        return


    def get_all_coordinates(self):
        '''
        This function takes no input and outputs the complete string with the coordinates
        of all the objects in view in the current image
        '''
        main_text_coordinates = '' # Initialize the variable where we'll store the coordinates
        for i, objct in enumerate(self.objects): # Loop through all of the objects
            print("     On object:", objct)
            b_box = self.find_bounding_box(objct) # Get current object's coordinates
            if b_box: # If find_bounding_box() doesn't return None
                class_index = self.obj_names.index( self.objects_label[i] )
                print("         Initial coordinates:", b_box, "   Class index: " , class_index)
                text_coordinates = self.format_coordinates(b_box, class_index) # Reformat coordinates to YOLOv3 format
                print("         YOLO-friendly coordinates:", text_coordinates)
                main_text_coordinates = main_text_coordinates + text_coordinates # Update main_text_coordinates variables whith each
                                                                                 # line corresponding to each class in the frame of the current image
            else:
                print("         Object not visible")
                pass

        return main_text_coordinates # Return all coordinates



    def format_coordinates(self, coordinates, classe):
        '''
        This function takes as inputs the coordinates created by the find_bounding box() function, the current class,
        the image width and the image height and outputs the coordinates of the bounding box of the current class
        '''
        # If the current class is in view of the camera
        if coordinates: 
            ## Change coordinates reference frame
            x1 = (coordinates[0][0])
            x2 = (coordinates[1][0])
            y1 = (1 - coordinates[1][1])
            y2 = (1 - coordinates[0][1])

            ## Get final bounding box information
            width = (x2-x1)  # Calculate the absolute width of the bounding box
            height = (y2-y1) # Calculate the absolute height of the bounding box
            # Calculate the absolute center of the bounding box
            cx = x1 + (width/2) 
            cy = y1 + (height/2)

            ## Formulate line corresponding to the bounding box of one class
            txt_coordinates = str(classe) + ' ' + str(cx) + ' ' + str(cy) + ' ' + str(width) + ' ' + str(height) + '\n'

            return txt_coordinates
        # If the current class isn't in view of the camera, then pass
        else:
            pass


    def create_report_file(self):
        return


    def update_report_file(self):
        #ureport.write('Progress: ' + str(render_counter) + ' Rotation: ' + str(axis_rotation) + ' z_d: ' + str(d / 10) + '\n')
        return


    def find_bounding_box(self, obj):
        """
        Returns camera space bounding box of the mesh object.

        Gets the camera frame bounding box, which by default is returned without any transformations applied.
        Create a new mesh object based on self.carre_bleu and undo any transformations so that it is in the same space as the
        camera frame. Find the min/max vertex coordinates of the mesh visible in the frame, or None if the mesh is not in view.

        :param scene:
        :param camera_object:
        :param mesh_object:
        :return:
        """

        """ Get the inverse transformation matrix. """
        matrix = self.camera.matrix_world.normalized().inverted()
        """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
        mesh = obj.to_mesh(preserve_all_data_layers=True)
        mesh.transform(obj.matrix_world)
        mesh.transform(matrix)

        """ Get the world coordinates for the camera frame bounding box, before any transformations. """
        frame = [-v for v in self.camera.data.view_frame(scene=self.scene)[:3]]

        lx = []
        ly = []

        for v in mesh.vertices:
            co_local = v.co
            z = -co_local.z

            if z <= 0.0:
                """ Vertex is behind the camera; ignore it. """
                continue
            else:
                """ Perspective division """
                frame = [(v / (v.z / z)) for v in frame]

            min_x, max_x = frame[1].x, frame[2].x
            min_y, max_y = frame[0].y, frame[1].y

            x = (co_local.x - min_x) / (max_x - min_x)
            y = (co_local.y - min_y) / (max_y - min_y)

            lx.append(x)
            ly.append(y)


        """ Image is not in view if all the mesh verts were ignored """
        if not lx or not ly:
            return None

        min_x = np.clip(min(lx), 0.0, 1.0)
        min_y = np.clip(min(ly), 0.0, 1.0)
        max_x = np.clip(max(lx), 0.0, 1.0)
        max_y = np.clip(max(ly), 0.0, 1.0)

        """ Image is not in view if both bounding points exist on the same side """
        if min_x == max_x or min_y == max_y:
            return None

        """ Figure out the rendered image size """
        render = self.scene.render
        fac = render.resolution_percentage * 0.01
        dim_x = render.resolution_x * fac
        dim_y = render.resolution_y * fac
        
        ## Verify there's no coordinates equal to zero
        coord_list = [min_x, min_y, max_x, max_y]
        if min(coord_list) == 0.0:
            indexmin = coord_list.index(min(coord_list))
            coord_list[indexmin] = coord_list[indexmin] + 0.0000001

        return (min_x, min_y), (max_x, max_y)
        
        

    def render_blender(self, count_f_name, suffix=''):
        # Define random parameters
        random.seed(random.randint(1,1000))
        self.xpix = random.randint(500, 1000)
        self.ypix = random.randint(500, 1000)
        self.xpix = 640
        self.ypix = 640
        self.percentage = random.randint(100, 100)
        self.samples = random.randint(25, 50)
        # Render images
        image_name = str(count_f_name) + '-' + suffix +'.png'
        self.export_render(self.xpix, self.ypix, self.percentage, self.samples, self.images_filepath, image_name)

    def export_render(self, res_x, res_y, res_per, samples, file_path, file_name):
        # Set all scene parameters
        bpy.context.scene.cycles.samples = samples
        self.scene.render.resolution_x = res_x
        self.scene.render.resolution_y = res_y
        self.scene.render.resolution_percentage = res_per
        self.scene.render.filepath =  file_path + '/' + file_name

        # Take picture of current visible scene
        bpy.ops.render.render(write_still=True)



    def blender_api_preliminary_initialization_to_be_sure(self):  
        # Select an active object by default at the beginning to avoid context errors
        default_active_object = bpy.data.objects.get('SL_30_KMH')
        if default_active_object:
            bpy.context.view_layer.objects.active = default_active_object 
        return
        


    def create_objects(self):  # This function creates a list of all the <bpy.data.objects>            
        result = []
        classe = []

        for obj in bpy.data.objects:
            for substring in self.obj_names:
            # Check if any substring from listB is present in stringA
                if substring in obj.name:
                    result.append(obj)
                    classe.append(substring)
                
        return result, classe


    def update_scene(self):
        ## Configure lighting
        energy1 = random.randint(4, 30) # Grab random light intensity
        self.light_1.data.energy = energy1 # Update the <bpy.data.objects['Light']> energy information
        energy2 = random.randint(4, 20) # Grab random light intensity
        self.light_2.data.energy = energy2 # Update the <bpy.data.objects['Light2']> energy information
        ## Projectors
        self.edit_shadow_projector_parameters(self.projector_noise_01)
        
        # Change environment
        self.change_background_hdri_image( self._hdri_images_path, random.uniform(0.5, 1.0) )

        # Signs effects
        self.edit_snow_effect(gain=0.3)
        self.edit_dirt_effect()
        
        return



    def main_rendering_loop(self):
        '''
        This function represent the main algorithm explained in the Tutorial, it accepts the
        rotation step as input, and outputs the images and the labels to the above specified locations.
        '''
        print('Number of renders to create:', self._max_simulation_steps)
        
        self.blender_api_preliminary_initialization_to_be_sure()
    
        ## create the dataset output folder
        if self._is_output_enabled:
            self.create_output_folder( self._output_path , self._nickname )        
            self.create_yolo_labels_file()       
        
        ## Dispose the traffic signs in the scene
        self.dispose_objects(sign_distance=5)
        self.set_camera()
        ## Define a counter to name each .png and .txt files that are outputted
        render_counter = 0      
        ## Create .txt file that record the progress of the data generation
        self.create_report_file()

        ### Simulation core here
        for obj in bpy.data.collections.get('Generated Scene').objects:
            for i in range(0, 20):

                ## Udpate counter
                render_counter += 1 

                ## Update camera position
                self.axis.location = obj.location 
                
                # Display demo information - Location of the camera
                print("On render:", render_counter)

                ##Configure simulation environment!! ---------
                self.update_scene()

                if self._is_output_enabled:
                    self.create_sample(render_counter)

                ## Show progress on batch of renders
                print('Progress =', render_counter, '/', self._max_simulation_steps )
        
                self.update_report_file()
                
                ## Stop the simulation by a hardcoded limit of iterations. suggested for testing
                if render_counter >= self._max_simulation_steps and self._max_simulation_steps > 0:
                    print('Reached maximum step number!!!!')
                    return
    
        ## Close the .txt file corresponding to the report
        #report.close()



## Run data generation
if __name__ == '__main__':
    # Initialize rendering class as r
    r = Render()

    r.max_simulation_steps = 500
    r.nickname = 'adversarial'
    r.is_output_enabled = True
    r.initial_camera_location = (0, -2.5, -0.2)

    # gio linux
    r.hdri_images_path = '/home/gionji/MDU/Blender/projects/syntetic_dataset_blender_generator_trafficsigns/textures/environment/'
    r.output_path = '/home/gionji/MDU/Datasets/blender_generated/'

    # Initialize camera
    r.set_camera()

    # Begin data generation
    r.main_rendering_loop()
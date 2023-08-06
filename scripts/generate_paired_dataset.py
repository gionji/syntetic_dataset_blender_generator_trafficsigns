## Import all relevant libraries
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

        self.max_simulation_steps = 10
        self.accept_render = 'y'
        self.nickname = 'paired_dataset'

        self.initial_camera_location = (0, -5, 3)

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
        #self.projector_noise_02 = bpy.data.objects['NoiseProjector.002']
        
        
        self.obj_names = [ 'SL_10_KMH', 'SL_15_KMH', 'SL_20_KMH', 'SL_25_KMH', 'SL_30_KMH', 'SL_35_KMH', 
                           'SL_40_KMH', 'SL_45_KMH', 'SL_50_KMH', 'SL_55_KMH', 'SL_60_KMH', 'SL_65_KMH',
                           'SL_70_KMH', 'SL_75_KMH', 'SL_80_KMH', 'SL_90_KMH', 'SL_100_KMH', 'SL_110_KMH',
                           'SL_120_KMH', 'SL_130_KMH']
                           
        self.objects, self.objects_label = self.create_objects() # Create list of bpy.data.objects from bpy.data.objects[1] to bpy.data.objects[N]


        ## Render information
        self.camera_d_limits = [0, 10] # Define range of heights z in m that the camera is going to pan through
        self.beta_limits     = [5, -15] # Define range of beta angles that the camera is going to pan through
        self.gamma_limits    = [-60, 60] # Define range of gamma angles that the camera is going to pan through
        
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
        
        # gio linux
        self.hdri_images_path = '/home/gionji/MDU/Blender/projects/syntetic_dataset_blender_generator_trafficsigns/textures/environment/'
        self.output_path = '/home/gionji/MDU/Datasets/blender_generated/'

        self.tick = 0
        self.tack = 0
        self.max_render_time = 0
        

    ## Helpers methods
    def create_output_folder(self, path, nickname):
        current_datetime = datetime.now()
        folder_name = current_datetime.strftime("%Y_%m_%d-%H_%M") + "-" + nickname
        folder_path = os.path.join(path, folder_name)
        
        self.images_filepath = folder_path
        self.labels_filepath = folder_path + '/Labels'        

        # Check if the folder doesn't exist
        if not os.path.exists(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            os.makedirs(folder_path + '/Labels' )
            print("Folder created:", folder_path)
        else:
            print("Folder already exists:", folder_path)
        return
    
            
    def create_yolo_labels_file(self):
        file_path = self.labels_filepath + "/labels.txt"
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


## Scenes methods               

    #define the relative position of the camera from the empty axis
    def set_camera(self):
        self.axis.rotation_euler = (0, 0, 0)
        self.axis.location = (0, 20, 0)
        self.camera.location = self.initial_camera_location


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
    

    
    def edit_snow_effect(self, gain=0):        
        #iterate on all SIGNS materials
        # Get all materials in the scene
        materials = bpy.data.materials

        # Iterate through each material
        for material in materials:
            # Check if the material has node-based shading
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
                bpy.data.node_groups["Snow and Ice Group"].nodes["Snowflake depth"].color_ramp.elements[0].position = random.uniform(0.2, 0.6) 
                bpy.data.node_groups["Snow and Ice Group"].nodes["Snowflake depth"].color_ramp.elements[1].position = random.uniform(0.2, 0.6) 

                ## effetto fico brina
                bpy.data.node_groups["Snow and Ice Group"].nodes["snow_ice_bsdf"].inputs[5].default_value = 0.736364



                bpy.context.view_layer.update()        

        return
    
    
    def edit_dirt_effect(self):
        bpy.data.node_groups["Surface Effects Group"].nodes["dirt_effect_ratio"].outputs[0].default_value = random.uniform(0.90, 1.0)    
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
                y = random.uniform(35.0, 45.0)
                z = random.uniform(0.0, 4.0)
                # Set the position of the copied sign
                copied_obj.location = (x, y, z)
            # Update the scene
            bpy.context.view_layer.update()
        self.objects, self.objects_label = self.create_objects() # Create list of bpy.data.objects from bpy.data.objects[1] to bpy.data.objects[N]       
        return


    def update_scene(self):
        ## Configure lighting
        energy1 = random.randint(4, 30) # Grab random light intensity
        self.light_1.data.energy = energy1 # Update the <bpy.data.objects['Light']> energy information
        energy2 = random.randint(4, 20) # Grab random light intensity
        self.light_2.data.energy = energy2 # Update the <bpy.data.objects['Light2']> energy information
        # Change environment
        self.edit_shadow_projector_parameters(self.projector_noise_01)
        self.change_background_hdri_image( self.hdri_images_path, random.uniform(0.5, 1.0) )
        self.edit_snow_effect()
        self.edit_dirt_effect()
        
        return



    def main_rendering_loop(self, rot_step):
        '''
        This function represent the main algorithm explained in the Tutorial, it accepts the
        rotation step as input, and outputs the images and the labels to the above specified locations.
             '''
        ## Calculate the number of images and labels to generate
        n_renders = self.calculate_n_renders(rot_step) # Calculate number of images
        print('Number of renders to create:', n_renders)
        
        # Select an active object by default at the beginning to avoid context errors
        default_active_object = bpy.data.objects.get('SL_30_KMH')
        if default_active_object:
            bpy.context.view_layer.objects.active = default_active_object     
        ## prechecks
        #self.list_materials_nodes_and_inputs()

        ## create the dataset output folder
        self.create_output_folder( self.output_path , self.nickname )
        self.create_yolo_labels_file()       
        
        ## Dispose the traffic signs in the scene
        self.dispose_objects(sign_distance=5)      
        
        # update the environment 
        self.update_scene()
        
        #accept_render = input('\nContinue?[Y/N]:  ') # Ask whether to procede with the data generation
        accept_render = self.accept_render


        if accept_render == 'Y' or accept_render == 'y': # If the user inputs 'Y' then procede with the data generation
            # Create .txt file that record the progress of the data generation
            #report_file_path = self.labels_filepath + '/progress_report.txt'
            #report = open(report_file_path, 'w')
            
            # Multiply the limits by 10 to adapt to the for loop
            dmin = int(self.camera_d_limits[0] * 10)
            dmax = int(self.camera_d_limits[1] * 10)
            
            # Define a counter to name each .png and .txt files that are outputted
            render_counter = 0
            
            # Define the step with which the pictures are going to be taken
            rotation_step = rot_step

            # Begin nested loops
            for d in range(dmin, dmax + 1, 4): # Loop to vary the height of the camera
                ## Update the height of the camera
                camera_y = - (abs(self.initial_camera_location[1]) + d/10)
                self.camera.location = (0, -(5+d/10), 3)  
                #self.camera.location = (0, 0, d/10)  # Divide the distance z by 10 to re-factor current height

                # Refactor the beta limits for them to be in a range from 0 to 360 to adapt the limits to the for loop
                min_beta = (-1)*self.beta_limits[0] + 90
                max_beta = (-1)*self.beta_limits[1] + 90

                for beta in range(min_beta, max_beta + 1, rotation_step): # Loop to vary the angle beta
                    beta_r = (-1)*beta + 90 # Re-factor the current beta

                    for gamma in range(self.gamma_limits[0], self.gamma_limits[1] + 1, rotation_step): # Loop to vary the angle gamma
                        render_counter += 1 # Update counter
                        
                        ## Update the rotation of the axis
                        axis_rotation = (m.radians(beta_r), 0, m.radians(gamma)) 
                        self.axis.rotation_euler = axis_rotation # Assign rotation to <bpy.data.objects['Empty']> object
                        
                        # Display demo information - Location of the camera
                        print("On render:", render_counter)
                        print("--> Location of the camera:")
                        print("     d:", d/10, "m")
                        print("     Beta:", str(beta_r)+"Degrees")
                        print("     Gamma:", str(gamma)+"Degrees")

                        ##Configure simulation environment!! --------------------------------------------------------------
                        self.update_scene()
                        ##---------------------------------------------------------------------------------------------------
                            
                        ## Generate render
                        self.edit_snow_effect(gain=0)                        
                        self.render_blender(render_counter, 'clear') # Take photo of current scene and ouput the render_counter.png file
                        
                        self.change_background_hdri_image( self.hdri_images_path +'/snowy/', random.uniform(0.5, 1.0) )
                        self.edit_snow_effect(gain=0.5)  
                        self.render_blender(render_counter, 'snow')
                    
                        # Display demo information - Photo information
                        #print("--> Picture information:")
                        #print("     Resolution:", (self.xpix*self.percentage, self.ypix*self.percentage))
                        #print("     Rendering samples:", self.samples)

                        ## Output Labels
                        self.write_annotation_file(render_counter)
 
                        ## Show progress on batch of renders
                        print('Progress =', str(render_counter) + '/' + str(n_renders))
                        #ureport.write('Progress: ' + str(render_counter) + ' Rotation: ' + str(axis_rotation) + ' z_d: ' + str(d / 10) + '\n')
                        
                        ## Stop the simulation by a hardcoded limit of iterations. suggested for testing
                        if render_counter >= self.max_simulation_steps and self.max_simulation_steps > 0:
                            print('Reached maximum step number!!!!')
                            return

            #report.close() # Close the .txt file corresponding to the report

        else: # If the user inputs anything else, then abort the data generation
            print('Aborted rendering operation')
            pass


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

    def calculate_n_renders(self, rotation_step):
        zmin = int(self.camera_d_limits[0] * 10)
        zmax = int(self.camera_d_limits[1] * 10)

        render_counter = 0
        rotation_step = rotation_step

        for d in range(zmin, zmax+1, 2):
            camera_location = (0,0,d/10)
            min_beta = (-1)*self.beta_limits[0] + 90
            max_beta = (-1)*self.beta_limits[1] + 90

            for beta in range(min_beta, max_beta+1,rotation_step):
                beta_r = 90 - beta

                for gamma in range(self.gamma_limits[0], self.gamma_limits[1]+1,rotation_step):
                    render_counter += 1
                    axis_rotation = (beta_r, 0, gamma)

        return render_counter



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


def handle_events(scene):
    for event in bpy.context.events:
        if event.type == 'Q' and event.value == 'PRESS':
            print("Script execution interrupted by user.")
            sys.exit(0)


## Run data generation
if __name__ == '__main__':
    # Initialize rendering class as r
    r = Render()
    # Initialize camera
    r.set_camera()
    # Begin data generation
    rotation_step = 10
    r.main_rendering_loop(rotation_step)
    
    

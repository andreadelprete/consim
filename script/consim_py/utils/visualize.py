import numpy as np
import pinocchio as pin 
from pinocchio.shortcuts import buildModelsFromUrdf, createDatas
from pinocchio.utils import npToTuple
from os.path import dirname, exists, join
import os, sys
import gepetto.corbaserver

try:
    import hppfcl
    WITH_HPP_FCL_BINDINGS = True
except:
    WITH_HPP_FCL_BINDINGS = False

def getModelPath(subpath, printmsg=False):
    paths = [
        join(dirname(dirname(dirname(dirname(__file__)))), 'robots'),
        join(dirname(dirname(dirname(__file__))), 'robots')
    ]
    try:
        from .path import EXAMPLE_ROBOT_DATA_MODEL_DIR
        paths.append(EXAMPLE_ROBOT_DATA_MODEL_DIR)
    except ImportError:
        pass
    paths += [join(p, '/opt/openrobots/share/example-robot-data/robots') for p in sys.path]
    for path in paths:
        if exists(join(path, subpath.strip('/'))):
            if printmsg:
                print("using %s as modelPath" % path)
            return path
    raise IOError('%s not found' % subpath)

def getVisualPath(modelPath):
    return join(modelPath, '../..')


class Visualizer(object):
    def __init__(self, windowName="consim_Window", sceneName="world", cameraTF=None):
        """ initialze gepetto viewer, loads gui and displays the sceene """
        try:
            self.viewer = gepetto.corbaserver.Client()
            gui = self.viewer.gui
            # Create window
            window_l = gui.getWindowList()
            if not windowName in window_l:
                self.windowID = self.viewer.gui.createWindow(windowName)
            else:
                self.windowID = self.viewer.gui.getWindowID(windowName)
            # Create scene if needed
            scene_l = gui.getSceneList()
            if sceneName not in scene_l:
                gui.createScene(sceneName)
            self.sceneName = sceneName
            gui.addSceneToWindow(sceneName, self.windowID)
        except:
            import warnings
            msg = ("Error while starting the viewer client.\n"
                   "Check whether gepetto-viewer is properly started"
                  )
            warnings.warn(msg, category=UserWarning, stacklevel=2)

        self.floorGroup = "world/floor"
        self.backgroundColor = [1.,1.,1.,1.]
        self.floorScale = [2., 2., 2.]
        self.floorColor = [0.7, 0.7, 0.7, 1.]
        self.cameraTF = cameraTF 

        if self.cameraTF is not None:
            self.viewer.gui.setCameraTransform(self.windowID, cameraTF)
        

        self.viewer.gui.setBackgroundColor1(self.windowID, self.backgroundColor)
        self.viewer.gui.setBackgroundColor2(self.windowID, self.backgroundColor)

        self.enableFloor = True 
        if self.enableFloor:
            self.viewer.gui.createGroup(self.floorGroup)
            self.viewer.gui.addFloor(self.floorGroup + "/flat")
            self.viewer.gui.setScale(self.floorGroup + "/flat", self.floorScale)
            self.viewer.gui.setColor(self.floorGroup + "/flat", self.floorColor)
            self.viewer.gui.setLightingMode(self.floorGroup + "/flat", "OFF")




class ConsimVisual(object):
    def __init__(self, name, filename, package_dirs, root_joint, consimVisualizer, visualOptions):
        self.name = name 
        self.contactNames= visualOptions["contact_names"] 
        self.robotColor = visualOptions["robot_color"]
        self.forceColor = visualOptions["force_color"]
        self.coneColor = visualOptions["cone_color"]

        self.urdfDir = filename
        self.meshDir = package_dirs
        self.viewer = consimVisualizer.viewer
        self.sceneName = consimVisualizer.sceneName
        model, collision_model, visual_model = buildModelsFromUrdf(filename, package_dirs, root_joint)
        self.model = model
        self.collision_model = collision_model
        self.visual_model = visual_model 
        self.display_collisions = False 
        self.display_visuals    = True 
        self.display_forces = True
        self.display_cones = True 
        self.data, self.collision_data, self.visual_data = createDatas(model,collision_model,visual_model)

        self.rootNodeName="pinocchio"
        self.viewerRootNodeName = self.sceneName + "/" + self.rootNodeName+ "_" + self.name

        self.forceGroup = self.viewerRootNodeName + "/contact_forces"
        self.frictionGroup = self.viewerRootNodeName + "/friction_cone" 

        self.x_axis = np.array([1., 0., 0.])
        self.z_axis = np.array([0., 0., 1.])

        self.totalWeight = sum(m.mass for m in self.model.inertias) * np.linalg.norm(self.model.gravity.linear)


    def addCones(self):
        """ add cone visuals for all defined contacts """
        for cn in self.contactNames:
            self.viewer.gui.addCone(self.frictionGroup+"/"+cn, .05, .1, self.coneColor)
    
    def addForces(self):
        """ add force vector visuals for all defined contacts """
        for cn in self.contactNames:
            self.viewer.gui.addArrow(self.forceGroup+"/"+cn, .004, .025, self.forceColor)
            

    def forcePose(self, name, force):
        """ computes unit vectors describing the force and populates the pose matrix  """
        unit_force = force / np.linalg.norm(force)
        res = np.cross(self.x_axis, unit_force, axis=0)
        res_norm = np.linalg.norm(res)
        if res_norm <= 1.e-8:
            return np.eye(3)
        projection = np.dot(self.x_axis, unit_force)
        res_skew = pin.skew(res)
        rotation = np.eye(3) + res_skew + np.dot(res_skew, res_skew) * (1 - projection) / res_norm**2
        pose = self.data.oMf[self.model.getFrameId(name)] 
        return pin.SE3ToXYZQUATtuple(pin.SE3(rotation, pose.translation))


    def displayContact(self, name, force, visibility="ON"):
        if visibility =="OFF":
            self.viewer.gui.setVisibility(self.forceGroup + "/" + name, "OFF")
            self.viewer.gui.setVisibility(self.frictionGroup + "/" + name, "OFF")
        else:
            forcePose = self.forcePose(name, force)
            # force magnitude 
            forceMagnitude = np.linalg.norm(force) # / self.totalWeight
            forceName = self.forceGroup + "/" + name
            self.viewer.gui.setVector3Property(forceName, "Scale", [1. * forceMagnitude, 1., 1.])

            # force vector 
            self.viewer.gui.applyConfiguration(forceName, forcePose)
            self.viewer.gui.setVisibility(self.forceGroup + "/" + name, "ALWAYS_ON_TOP")
            # friction cone 
             
            # self.viewer.gui.applyConfiguration(self.frictionGroup + "/" + name, 
            # pin.SE3ToXYZQUATtuple(pose))
            # self.viewer.gui.setVisibility(self.frictionGroup + "/" + name, "ON")




    def getViewerNodeName(self, geometry_object, geometry_type):
        """Return the name of the geometry object inside the viewer"""
        if geometry_type is pin.GeometryType.VISUAL:
            return self.viewerVisualGroupName + '/' + geometry_object.name
        elif geometry_type is pin.GeometryType.COLLISION:
            return self.viewerCollisionGroupName + '/' + geometry_object.name
 


    def loadViewerGeometryObject(self, geometry_object, geometry_type):
        """Load a single geometry object"""

        gui = self.viewer.gui

        meshName = self.getViewerNodeName(geometry_object,geometry_type)
        meshPath = geometry_object.meshPath
        meshTexturePath = geometry_object.meshTexturePath
        meshScale = geometry_object.meshScale
        meshColor = geometry_object.meshColor

        try:
            if meshName == "":
                msg = "Display of geometric primitives is supported only if pinocchio is build with HPP-FCL bindings."
                warnings.warn(msg, category=UserWarning, stacklevel=2)
                return
            success = gui.addMesh(meshName, meshPath)
            if not success:
                return
        except Exception as e:
            msg = "Error while loading geometry object: %s\nError message:\n%s" % (geometry_object.name, e)
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return

        gui.setScale(meshName, npToTuple(meshScale))
        if geometry_object.overrideMaterial:
            gui.setColor(meshName, npToTuple(meshColor))
            if meshTexturePath != '':
                gui.setTexture(meshName, meshTexturePath)

    def loadViewerModel(self):
        """Create the scene displaying the robot meshes in gepetto-viewer"""
        gui = self.viewer.gui
    
        if not gui.nodeExists(self.viewerRootNodeName):
            gui.createGroup(self.viewerRootNodeName)

        self.viewerVisualGroupName = self.viewerRootNodeName + "/" + "visuals"
        if not gui.nodeExists(self.viewerVisualGroupName):
            gui.createGroup(self.viewerVisualGroupName)
            
        if self.visual_model is not None:
            for visual in self.visual_model.geometryObjects:
                self.loadViewerGeometryObject(visual,pin.GeometryType.VISUAL)
        self.displayVisuals(self.visual_model is not None)

        # create force and cone groups       
        self.viewer.gui.createGroup(self.forceGroup)  ### display active contact forces 
        self.viewer.gui.createGroup(self.frictionGroup)  ### display active friction cones 

        self.addForces()
        self.addCones()

        gui.refresh()

    def display(self, q):
        """Display the robot at configuration q in the viewer by placing all the bodies."""
        gui = self.viewer.gui
        # Update the robot kinematics and geometry.
        pin.framesForwardKinematics(self.model,self.data,q)
        #

        if self.display_visuals:
            pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
            gui.applyConfigurations (
                    [ self.getViewerNodeName(visual,pin.GeometryType.VISUAL) for visual in self.visual_model.geometryObjects ],
                    [ pin.SE3ToXYZQUATtuple(self.visual_data.oMg[self.visual_model.getGeometryId(visual.name)]) for visual in self.visual_model.geometryObjects ]
                    )


        for cn in self.contactNames:
            fvalue = np.array([0.,0.,5.]) # only a test for now
            forceVisiblity = "ON"
            self.displayContact(cn, fvalue, forceVisiblity) 


        gui.refresh()


    def displayVisuals(self,visibility):
        """Set whether to display visual objects or not"""
        gui = self.viewer.gui
        self.display_visuals = visibility
        if self.visual_model is None: return

        if visibility:
            visibility_mode = "ON"
        else:
            visibility_mode = "OFF"

        for visual in self.visual_model.geometryObjects:
            nodeName = self.getViewerNodeName(visual,pin.GeometryType.VISUAL)
            gui.setVisibility(nodeName,visibility_mode)
            gui.setColor(nodeName, self.robotColor)

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
    def __init__(self, name, filename, package_dirs, root_joint, consimVisualizer):
        self.name = name 
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
        self.data, self.collision_data, self.visual_data = createDatas(model,collision_model,visual_model)

        self.viewerRootNodeName = self.sceneName + "/" + rootNodeName+ "_" + self.name

        self.forceGroup = self.viewerRootNodeName + "/contact_forces"
        self.frictionGroup = self.viewerRootNodeName + "/friction_cone"


        self.viewer.gui.createGroup(self.forceGroup)  ### display active contact forces 
        self.viewer.gui.createGroup(self.frictionGroup)  ### display active friction cones 
    

    def addCones(self):
        """ add cone visuals for all defined contacts """
        pass 
    
    def addForces(self):
        """ add force vector visuals for all defined contacts """
        pass 

    def displayContact(self, name, force, visibility="ON"):
        if visibility =="OFF":
            self.viewer.gui.setVisibility(self.forceGroup + "/" + name, "OFF")
            self.viewer.gui.setVisibility(self.frictionGroup + "/" + name, "OFF")
        else:
            pose = self.data.oMf[self.model.getFrameId(name)] # get pose of current contact 
            # cone and force vector orientation 

            # cone scaling 

            # force vector scaling 


    def getViewerNodeName(self, geometry_object, geometry_type):
        """Return the name of the geometry object inside the viewer"""
        if geometry_type is pin.GeometryType.VISUAL:
            return self.viewerVisualGroupName + '/' + geometry_object.name
        elif geometry_type is pin.GeometryType.COLLISION:
            return self.viewerCollisionGroupName + '/' + geometry_object.name
 


    def loadPrimitive(self, meshName, geometry_object):

        gui = self.viewer.gui

        meshColor = geometry_object.meshColor

        geom = geometry_object.geometry
        if isinstance(geom, hppfcl.Capsule):
            return gui.addCapsule(meshName, geom.radius, 2. * geom.halfLength, npToTuple(meshColor))
        elif isinstance(geom, hppfcl.Cylinder):
            return gui.addCylinder(meshName, geom.radius, 2. * geom.halfLength, npToTuple(meshColor))
        elif isinstance(geom, hppfcl.Box):
            w, h, d = npToTuple(2. * geom.halfSide)
            return gui.addBox(meshName, w, h, d, npToTuple(meshColor))
        elif isinstance(geom, hppfcl.Sphere):
            return gui.addSphere(meshName, geom.radius, npToTuple(meshColor))
        elif isinstance(geom, hppfcl.Cone):
            return gui.addCone(meshName, geom.radius, 2. * geom.halfLength, npToTuple(meshColor))
        elif isinstance(geom, hppfcl.Convex):
            pts = [ npToTuple(geom.points(geom.polygons(f)[i])) for f in range(geom.num_polygons) for i in range(3) ]
            gui.addCurve(meshName, pts, npToTuple(meshColor))
            gui.setCurveMode(meshName, "TRIANGLES")
            gui.setLightingMode(meshName, "ON")
            gui.setBoolProperty(meshName, "BackfaceDrawing", True)
            return True
        elif isinstance(geom, hppfcl.ConvexBase):
            pts = [ npToTuple(geom.points(i)) for i in range(geom.num_points) ]
            gui.addCurve(meshName, pts, npToTuple(meshColor))
            gui.setCurveMode(meshName, "POINTS")
            gui.setLightingMode(meshName, "OFF")
            return True
        else:
            msg = "Unsupported geometry type for %s (%s)" % (geometry_object.name, type(geom) )
            warnings.warn(msg, category=UserWarning, stacklevel=2)
            return False

    def loadViewerGeometryObject(self, geometry_object, geometry_type):
        """Load a single geometry object"""

        gui = self.viewer.gui

        meshName = self.getViewerNodeName(geometry_object,geometry_type)
        meshPath = geometry_object.meshPath
        meshTexturePath = geometry_object.meshTexturePath
        meshScale = geometry_object.meshScale
        meshColor = geometry_object.meshColor

        try:
            if WITH_HPP_FCL_BINDINGS and isinstance(geometry_object.geometry, hppfcl.ShapeBase):
                success = self.loadPrimitive(meshName, geometry_object)
            else:
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

    def loadViewerModel(self, rootNodeName="pinocchio"):
        """Create the scene displaying the robot meshes in gepetto-viewer"""

        # Start a new "scene" in this window, named "world", with just a floor.
        gui = self.viewer.gui
        

        if not gui.nodeExists(self.viewerRootNodeName):
            gui.createGroup(self.viewerRootNodeName)

        self.viewerCollisionGroupName = self.viewerRootNodeName + "/" + "collisions"
        if not gui.nodeExists(self.viewerCollisionGroupName):
            gui.createGroup(self.viewerCollisionGroupName)

        self.viewerVisualGroupName = self.viewerRootNodeName + "/" + "visuals"
        if not gui.nodeExists(self.viewerVisualGroupName):
            gui.createGroup(self.viewerVisualGroupName)

        # iterate over visuals and create the meshes in the viewer
        if self.collision_model is not None:
            for collision in self.collision_model.geometryObjects:
                self.loadViewerGeometryObject(collision,pin.GeometryType.COLLISION)
        # Display collision if we have them and there is no visual
        self.displayCollisions(self.collision_model is not None and self.visual_model is None)

        if self.visual_model is not None:
            for visual in self.visual_model.geometryObjects:
                self.loadViewerGeometryObject(visual,pin.GeometryType.VISUAL)
        self.displayVisuals(self.visual_model is not None)

        # Finally, refresh the layout to obtain your first rendering.
        gui.refresh()

    def display(self, q):
        """Display the robot at configuration q in the viewer by placing all the bodies."""
        gui = self.viewer.gui
        # Update the robot kinematics and geometry.
        pin.forwardKinematics(self.model,self.data,q)
        #
        if self.display_collisions:
                pin.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data)
                gui.applyConfigurations (
                        [ self.getViewerNodeName(collision,pin.GeometryType.COLLISION) for collision in self.collision_model.geometryObjects ],
                        [ pin.SE3ToXYZQUATtuple(self.collision_data.oMg[self.collision_model.getGeometryId(collision.name)]) for collision in self.collision_model.geometryObjects ]
                        )

        if self.display_visuals:
            pin.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
            gui.applyConfigurations (
                    [ self.getViewerNodeName(visual,pin.GeometryType.VISUAL) for visual in self.visual_model.geometryObjects ],
                    [ pin.SE3ToXYZQUATtuple(self.visual_data.oMg[self.visual_model.getGeometryId(visual.name)]) for visual in self.visual_model.geometryObjects ]
                    )

        gui.refresh()


    def displayCollisions(self,visibility):
        """Set whether to display collision objects or not"""
        gui = self.viewer.gui
        self.display_collisions = visibility
        if self.collision_model is None: return

        if visibility:
            visibility_mode = "ON"
        else:
            visibility_mode = "OFF"

        for collision in self.collision_model.geometryObjects:
            nodeName = self.getViewerNodeName(collision,pin.GeometryType.COLLISION)
            gui.setVisibility(nodeName,visibility_mode)



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


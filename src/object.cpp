
#include "consim/object.hpp"

namespace consim {

ContactObject::ContactObject(const std::string & name, ContactModel& contact_model):
    name_(name), contact_model_(&contact_model) { }

// -------------------------------------------------------------------------------

bool FloorObject::checkCollision(ContactPoint &cp)
{
  // checks for penetration into the floor 
  if (cp.x(2) > 0.) {
    return false;
  }

  if (!cp.active) {
    cp.x_anchor = cp.x;
    cp.v_anchor.setZero();
    cp.predictedX0_ = cp.x;
    cp.contactNormal_ << 0.,0.,1.; 
    cp.contactTangentA_ << 0.,1.,0.;
    cp.contactTangentB_ << 1.,0.,0.;
  }
  //
  return true;
}

void FloorObject::computePenetration(ContactPoint &cp){
  /** compute displacement relative to contact object
   * delta_x: relative penetration 
   * normal: penetration along normal to contact object
   * tangent: penetration along tangent to contact object
   * normalvel: velocity along normal to contact object
   * tanvel: velocity along tangent to contact object
   * */ 
  cp.delta_x = cp.x_anchor - cp.x; 
  cp.normal = cp.delta_x.dot(cp.contactNormal_) * cp.contactNormal_; 
  cp.tangent = cp.delta_x - cp.normal; 
  cp.normvel = (cp.v).dot(cp.contactNormal_) * cp.contactNormal_; 
  cp.tanvel = cp.v - cp.normvel; 
}


HalfPlaneObject::HalfPlaneObject(const std::string & name, ContactModel& contact_model, double alpha)
  : ContactObject(name, contact_model), angle_(alpha)
  { 

    /**
     * given a slope angle, an axis of rotation, then rotate the basis for the plane
     * a plane equation is described as  ax + by + cz + d = 0 
     * the normal is given by n = [a,b,c]
     * and d = -a x_0 -b y_0 - c z_0 
     * where (x_0 , y_0 ,z_0) is any point on the plane 
     * the signed distance to the plane is then given by  D =   (a x_i + b y_i + c z_i + d)/ norm(plane_normal)  
     *  
    **/ 
    // brief\ for now exclusively rotation around y and intersects the Zero z-axis at zero x and y 
    t_ = Eigen::AngleAxisd(angle_, Eigen::Vector3d(0.,1.,0.));

    Eigen::Vector3d flatNormal;   flatNormal<< 0.,0.,1.; 
    Eigen::Vector3d flatTangentA; flatTangentA<< 0.,1.,0.;
    Eigen::Vector3d flatTangentB; flatTangentB<< 1.,0.,0.;

    Eigen::Vector3d plane_point;  plane_point<< 0.,0.,0.; 

    planeNormal_   = t_.linear() * flatNormal;
    planeTangentA_ = t_.linear() * flatTangentA;
    planeTangentB_ = t_.linear() * flatTangentB;

    plane_offset_ = - planeNormal_.transpose()*plane_point;  // d in the plane equation above 
    

  };

void HalfPlaneObject::computeDistance(ContactPoint &cp){
  distance_ = planeNormal_.transpose()*cp.x;  
  distance_ += plane_offset_;  
}

bool HalfPlaneObject::checkCollision(ContactPoint &cp)
{
  // checks for penetration into the plane 
  computeDistance(cp);
  
  if (distance_ > 0.) {
    return false;
  }

  if (!cp.active) {
    cp.x_anchor = cp.x;
    cp.v_anchor.setZero();
    cp.predictedX0_ = cp.x;
    cp.contactNormal_ = planeNormal_;  
    cp.contactTangentA_ = planeTangentA_;
    cp.contactTangentB_ = planeTangentB_;
  }
  //
  return true;
}

void HalfPlaneObject::computePenetration(ContactPoint &cp){
  /** compute displacement relative to contact object
   * delta_x: relative penetration 
   * normal: penetration along normal to contact object
   * tangent: penetration along tangent to contact object
   * normalvel: velocity along normal to contact object
   * tanvel: velocity along tangent to contact object
   * */ 
  cp.delta_x = cp.x_anchor - cp.x; 
  cp.normal = cp.delta_x.dot(cp.contactNormal_) * cp.contactNormal_; 
  cp.tangent = cp.delta_x - cp.normal; 
  cp.normvel = (cp.v).dot(cp.contactNormal_) * cp.contactNormal_; 
  cp.tanvel = cp.v - cp.normvel; 
}

}

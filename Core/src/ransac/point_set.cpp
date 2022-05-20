
#include "point_set.h"	

#include <algorithm>



PointSet::PointSet() {
}


PointSet::~PointSet() {
	clear();
}


Box3 PointSet::bounding_box_dirty() const {
	return Geom::bounding_box(this);
}


std::vector<vec3>& PointSet::normals(bool alloc) {
	std::vector<vec3>& nms = vector_attributes_["normal"];
	if (alloc && nms.size() != points_.size())
		nms.resize(points_.size());
	return nms;
}


const std::vector<vec3>& PointSet::normals(bool alloc) const {
	std::vector<vec3>& nms = vector_attributes_["normal"];
	if (alloc && nms.size() != points_.size())
		nms.resize(points_.size());
	return nms;
}


std::vector<vec3>& PointSet::colors(bool alloc) {
	std::vector<vec3>& cls = vector_attributes_["color"];
	if (alloc && cls.size() != points_.size())
		cls.resize(points_.size());
	return cls;
}


const std::vector<vec3>& PointSet::colors(bool alloc) const {
	std::vector<vec3>& cls = vector_attributes_["color"];
	if (alloc && cls.size() != points_.size())
		cls.resize(points_.size());
	return cls;
}


bool PointSet::has_normals() const {
	return has_vector_attribute("normal");
}


bool PointSet::has_colors() const {
	return has_vector_attribute("color");
}


void PointSet::clear() 
{
	points_.clear();
	groups_.clear();

	scalar_attributes_.clear();
	vector_attributes_.clear();

	std::map< std::string, void*>::iterator pos = attributes_.begin();
	for (; pos != attributes_.end(); ++pos) {
		delete pos->second;
	}
	attributes_.clear();
}


bool PointSet::has_scalar_attribute(const std::string& name) const {
	std::map<std::string, std::vector<float>>::const_iterator pos = scalar_attributes_.find(name);
	if (pos != scalar_attributes_.end() && pos->second.size() == points_.size())
		return true;
	return false;
}


bool PointSet::has_vector_attribute(const std::string& name) const {
	std::map<std::string, std::vector<vec3>>::const_iterator pos = vector_attributes_.find(name);
	if (pos != vector_attributes_.end() && pos->second.size() == points_.size())
		return true;
	return false;
}


void PointSet::delete_scalar_attribute(const std::string& name) {
	std::map<std::string, std::vector<float>>::iterator pos = scalar_attributes_.find(name);
	if (pos == scalar_attributes_.end()) {
		Logger::warn(title()) << "scalar attribute \'" << name << "\' does not exist" << std::endl;
		return;
	}
	else
		scalar_attributes_.erase(pos);
}


void PointSet::delete_vector_attribute(const std::string& name) {
	std::map<std::string, std::vector<vec3>>::iterator pos = vector_attributes_.find(name);
	if (pos == vector_attributes_.end()) {
		Logger::warn(title()) << "vector attribute \'" << name << "\' does not exist" << std::endl;
		return;
	}
	else
		vector_attributes_.erase(pos);
}


std::vector<std::string> PointSet::named_scalar_attributes() const {
	std::vector<std::string> names;
	std::map<std::string, std::vector<float>>::const_iterator it = scalar_attributes_.begin();
	for (; it != scalar_attributes_.end(); ++it) {
		names.push_back(it->first);
	}
	return names;
}


std::vector<std::string> PointSet::named_vector_attributes() const {
	std::vector<std::string> names;
	std::map<std::string, std::vector<vec3>>::const_iterator it = vector_attributes_.begin();
	for (; it != vector_attributes_.end(); ++it) {
		names.push_back(it->first);
	}
	return names;
}


std::vector<float>& PointSet::scalar_attribute(const std::string& name, bool alloc) {
	std::map<std::string, std::vector<float>>::iterator pos = scalar_attributes_.find(name);
	if (pos != scalar_attributes_.end()) {
		if (alloc && pos->second.size() != points_.size())
			pos->second.resize(points_.size());
		return pos->second;
	}
	else {
		scalar_attributes_[name] = std::vector<float>();
		Logger::out(title()) << "scalar attribute \'" << name << "\' created" << std::endl;

		if (alloc && scalar_attributes_[name].size() != points_.size())
			scalar_attributes_[name].resize(points_.size());
		return scalar_attributes_[name];
	}
}


const std::vector<float>& PointSet::scalar_attribute(const std::string& name, bool alloc) const {
    return const_cast<PointSet*>(this)->scalar_attribute(name, alloc);
}


std::vector<vec3>&  PointSet::vector_attribute(const std::string& name, bool alloc) {
	std::map<std::string, std::vector<vec3>>::iterator pos = vector_attributes_.find(name);
	if (pos != vector_attributes_.end()) {
		if (alloc && pos->second.size() != points_.size())
			pos->second.resize(points_.size());
		return pos->second;
	}
	else {
		vector_attributes_[name] = std::vector<vec3>();
		Logger::out(title()) << "vector attribute \'" << name << "\' created" << std::endl;

		if (alloc && vector_attributes_[name].size() != points_.size())
			vector_attributes_[name].resize(points_.size());
		return vector_attributes_[name];
	}
}


const std::vector<vec3>&  PointSet::vector_attribute(const std::string& name, bool alloc) const {
    return const_cast<PointSet*>(this)->vector_attribute(name, alloc);
}


void PointSet::delete_points(const std::vector<int>& indices) {
	if (indices.empty())
		return;

	std::size_t old_num = points_.size();
	std::size_t new_num = points_.size() - indices.size();
	if (new_num <= 0) { // delete all points
		clear();
		return;
	}
	 
	//////////////////////////////////////////////////////////////////////////
	// mark the status for each point
	std::vector<unsigned char>  remained(points_.size(), 1);
	for (std::size_t i = 0; i < indices.size(); ++i) {
		int idx = indices[i];
		remained[idx] = 0;
	}

	//////////////////////////////////////////////////////////////////////////

	// step 1. clean vertices
	std::vector<vec3>	new_points(new_num);
	// since some points are deleted, the indices of the points will change
	std::vector<int>	new_index(old_num, -1);
	int idx = 0;
	for (std::size_t i = 0; i < remained.size(); ++i) {
		if (remained[i]) {
			new_points[idx] = points_[i];
			new_index[i] = idx;
			++idx;
		}
	}
	points_ = new_points;

	//////////////////////////////////////////////////////////////////////////

	// step 2. update vertex groups
	// since some points are deleted, the indices in the group should change accordingly
	if (!groups_.empty()) {
		std::vector<VertexGroup::Ptr> new_groups; // some groups may become empty so will be removed.
		for (std::size_t i = 0; i < groups_.size(); ++i) {
			VertexGroup::Ptr g = groups_[i];
			std::vector<int> new_g;
			for (std::size_t j = 0; j < g->size(); ++j) {
				int old_id = g->at(j);
				if (remained[old_id]) {
					int new_id = new_index[old_id];
					new_g.push_back(new_id);
				}
			}

			if (new_g.size() > 0) {
				g->clear();  // clear original indices
				g->insert(g->end(), new_g.begin(), new_g.end());
				new_groups.push_back(g);
			}
		}
		groups_ = new_groups;
	}

	//////////////////////////////////////////////////////////////////////////

	// step 3. attributes
	std::map<std::string, std::vector<float>>::iterator sit = scalar_attributes_.begin();
	for (; sit != scalar_attributes_.end(); ++sit) {
		std::vector<float>& attrib = sit->second;
		std::vector<float> new_attrib(new_num);
		for (std::size_t i = 0; i < attrib.size(); ++i) {
			if (remained[i]) {
				int idx = new_index[i];
				new_attrib[idx] = attrib[i];
			}
		}
		attrib = new_attrib;
	}

	std::map<std::string, std::vector<vec3>>::iterator vit = vector_attributes_.begin();
	for (; vit != vector_attributes_.end(); ++vit) {
		std::vector<vec3>& attrib = vit->second;
		std::vector<vec3> new_attrib(new_num);
		for (std::size_t i = 0; i < attrib.size(); ++i) {
			if (remained[i]) {
				int idx = new_index[i];
				new_attrib[idx] = attrib[i];
			}
		}
		attrib = new_attrib;
	}

	notify_vertex_change();
}


std::string PointSet::attribute_type_name(const std::string& name) const {
	std::map<std::string, std::string>::iterator pos = attribute_types_.find(name);
	if (pos == attribute_types_.end()) {
		std::cerr << "attribute \'" << name << "\' doesn't exist" << std::endl;
		return "error";
	}
	else {
		return AttributeSerializer::find_name_by_type(attribute_types_[name]);
	}
}

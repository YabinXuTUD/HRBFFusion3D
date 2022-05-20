
#ifndef _GEOMETRY_POINT_SET_H_
#define _GEOMETRY_POINT_SET_H_


#include "vertex_group.h"

#include <map>
#include <vector>
#include <string>

#include<Eigen/Core>

/*
*	Version:	 1.0
*	created:	 Dec. 14, 2015
*	author:		 Liangliang Nan
*	contact:     liangliang.nan@gmail.com
*/

// Since I would like to process huge scans (up to billions of points), an earlier 
// implementation based on double-connected list has issues in the follow aspects:
//  1) no random access of the data;
//  2) OpenGL renderring overhead (needs packing to transfer data into GPU);
//  3) hard to employ OMP support;
//  4) file management (reading and writing large blocks);
//  5) selection, etc.
// Thus I have this implementation using std::vector<vec3> for data storage.


class PointSet
{
public:
	PointSet();
	virtual ~PointSet();

	int  num_of_points() const  { return int(points_.size()); }

	bool has_normals() const;
	bool has_colors() const;

	// --------------- data access

        std::vector<Eigen::Vector3f>& points() { return points_; }
        const std::vector<Eigen::Vector3f>& points() const { return points_; }
        Eigen::Vector3f& point(std::size_t idx) { return points_[idx]; }
        const Eigen::Vector3f& point(std::size_t idx) const { return points_[idx]; }

        std::vector<Eigen::Vector3f>& normals(bool alloc = false);
        const std::vector<Eigen::Vector3f>& normals(bool alloc = false) const;


	//---------------- vertex groups

	std::vector<VertexGroup::Ptr>& groups() { return groups_; }
	const std::vector<VertexGroup::Ptr>& groups() const { return groups_; }

	// --------------- data management

	// currently I assume all indices are in the range [0, points_.size()]
	void delete_points(const std::vector<int>& indices);
	// delete all points
	void clear();

	//----------------- more attributes 
	
	bool has_scalar_attribute(const std::string& name) const;
	bool has_vector_attribute(const std::string& name) const;

	void delete_scalar_attribute(const std::string& name);
	void delete_vector_attribute(const std::string& name);

	std::vector<std::string> named_scalar_attributes() const;
	std::vector<std::string> named_vector_attributes() const;

	std::vector<float>& scalar_attribute(const std::string& name, bool alloc = false);				// find or create
	const std::vector<float>& scalar_attribute(const std::string& name, bool alloc = false) const;	// find or create
	
	std::vector<vec3>&  vector_attribute(const std::string& name, bool alloc = false);				// find or create
	const std::vector<vec3>&  vector_attribute(const std::string& name, bool alloc = false) const;	// find or create
	
protected:

	virtual Box3 bounding_box_dirty() const;

private:
        std::vector<Eigen::Vector3f>				points_;
	std::vector<VertexGroup::Ptr>	groups_;

	mutable std::map< std::string, std::vector<float> >	scalar_attributes_;
        mutable std::map< std::string, std::vector<Eigen::Vector3f>  >	vector_attributes_;

public:
	template <typename T>
	std::vector<T>& attribute(const std::string& name, bool alloc = false);

	template <typename T>
	const std::vector<T>& attribute(const std::string& name, bool alloc = false) const;

	std::string attribute_type_name(const std::string& name) const;

private:
	// maps of std::string can be super slow when calling find with a string literal or const char* 
	// as find forces construction/copy/destruction of a std::sting copy of the const char*.
	mutable std::map<std::string, void*>		attributes_;
	mutable std::map<std::string, std::string>	attribute_types_;
};




template <typename T>
std::vector<T>& PointSet::attribute(const std::string& name, bool alloc /* = false*/) {
	std::map< std::string, void* >::iterator pos = attributes_.find(name);
	if (pos != attributes_.end()) {
		std::vector<T>* attrib = reinterpret_cast<std::vector<T>*>(pos->second);
		if (alloc && attrib->size() != points_.size())
			attrib->resize(points_.size());
		return *attrib;
	}
	else {
		std::vector<T>* attrib = new std::vector<T>();
		attributes_[name] = attrib;
		attribute_types_[name] = typeid(T).name();
		if (alloc && attrib->size() != points_.size())
			attrib->resize(points_.size());
		return *attrib;
	}
}



template <typename T>
const std::vector<T>& PointSet::attribute(const std::string& name, bool alloc /* = false*/) const {
	std::map< std::string, void* >::iterator pos = attributes_.find(name);
	if (pos != attributes_.end()) {
		std::vector<T>* attrib = reinterpret_cast<std::vector<T>*>(pos->second);
		if (alloc && attrib->size() != points_.size())
			attrib->resize(points_.size());
		return *attrib;
	}
	else {
		std::vector<T>* attrib = new std::vector<T>();
		attributes_[name] = attrib;
		attribute_types_[name] = typeid(T).name();
		if (alloc && attrib->size() != points_.size())
			attrib->resize(points_.size());
		return *attrib;
	}
}

#endif

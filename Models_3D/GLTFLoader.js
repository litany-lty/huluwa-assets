import {
	BufferAttribute,
	BufferGeometry,
	ClampToEdgeWrapping,
	Color,
	DoubleSide,
	FileLoader,
	FrontSide,
	Group,
	ImageBitmapLoader,
	InterleavedBuffer,
	InterleavedBufferAttribute,
	LinearFilter,
	LinearMipmapLinearFilter,
	LinearMipmapNearestFilter,
	LinearSRGBColorSpace,
	Loader,
	LoaderUtils,
	Matrix4,
	Mesh,
	MeshBasicMaterial,
	MeshPhysicalMaterial,
	MeshStandardMaterial,
	MirroredRepeatWrapping,
	NearestFilter,
	NearestMipmapLinearFilter,
	NearestMipmapNearestFilter,
	NoColorSpace,
	Object3D,
	Quaternion,
	RepeatWrapping,
	SRGBColorSpace,
	Texture,
	TextureLoader,
	Vector2,
	Vector3,
} from 'three';

const GLB_MAGIC = 0x46546C67; // 'glTF'
const CHUNK_JSON = 0x4E4F534A; // 'JSON'
const CHUNK_BIN = 0x004E4942; // 'BIN\0'

const COMPONENT_TYPES = {
	5120: Int8Array,
	5121: Uint8Array,
	5122: Int16Array,
	5123: Uint16Array,
	5125: Uint32Array,
	5126: Float32Array,
};

const TYPE_SIZES = {
	SCALAR: 1,
	VEC2: 2,
	VEC3: 3,
	VEC4: 4,
	MAT2: 4,
	MAT3: 9,
	MAT4: 16,
};

const WEBGL_WRAPPINGS = {
	33071: ClampToEdgeWrapping,
	33648: MirroredRepeatWrapping,
	10497: RepeatWrapping,
};

const WEBGL_FILTERS = {
	9728: NearestFilter,
	9729: LinearFilter,
	9984: NearestMipmapNearestFilter,
	9985: LinearMipmapNearestFilter,
	9986: NearestMipmapLinearFilter,
	9987: LinearMipmapLinearFilter,
};

const ATTRIBUTE_MAP = {
	POSITION: 'position',
	NORMAL: 'normal',
	TANGENT: 'tangent',
	TEXCOORD_0: 'uv',
	TEXCOORD_1: 'uv1',
	COLOR_0: 'color',
	JOINTS_0: 'skinIndex',
	WEIGHTS_0: 'skinWeight',
};

export class GLTFLoader extends Loader {
	constructor( manager ) {
		super( manager );
	}

	load( url, onLoad, onProgress, onError ) {
		const resourcePath = LoaderUtils.extractUrlBase( url );
		this.manager.itemStart( url );

		const loader = new FileLoader( this.manager );
		loader.setPath( this.path );
		loader.setResponseType( 'arraybuffer' );
		loader.setRequestHeader( this.requestHeader );
		loader.setWithCredentials( this.withCredentials );

		loader.load(
			url,
			( data ) => {
				this.parse(
					data,
					resourcePath,
					( gltf ) => {
						if ( onLoad ) onLoad( gltf );
						this.manager.itemEnd( url );
					},
					( err ) => {
						if ( onError ) onError( err );
						this.manager.itemError( url );
						this.manager.itemEnd( url );
					}
				);
			},
			onProgress,
			( err ) => {
				if ( onError ) onError( err );
				this.manager.itemError( url );
				this.manager.itemEnd( url );
			}
		);
	}

	loadAsync( url, onProgress ) {
		return new Promise( ( resolve, reject ) => {
			this.load( url, resolve, onProgress, reject );
		} );
	}

	parse( data, path, onLoad, onError ) {
		try {
			let json = null;
			let binChunk = null;

			if ( data instanceof ArrayBuffer ) {
				const header = new DataView( data, 0, 12 );
				const magic = header.getUint32( 0, true );
				if ( magic === GLB_MAGIC ) {
					const totalLength = header.getUint32( 8, true );
					let offset = 12;
					const decoder = new TextDecoder();
					while ( offset < totalLength ) {
						const chunkView = new DataView( data, offset, 8 );
						const chunkLen = chunkView.getUint32( 0, true );
						const chunkType = chunkView.getUint32( 4, true );
						offset += 8;
						if ( chunkType === CHUNK_JSON ) {
							json = JSON.parse( decoder.decode( new Uint8Array( data, offset, chunkLen ) ) );
						} else if ( chunkType === CHUNK_BIN ) {
							binChunk = data.slice( offset, offset + chunkLen );
						}
						offset += chunkLen;
					}
				} else {
					json = JSON.parse( new TextDecoder().decode( new Uint8Array( data ) ) );
				}
			} else if ( typeof data === 'string' ) {
				json = JSON.parse( data );
			}

			if ( ! json ) {
				throw new Error( 'GLTFLoader: Failed to parse glTF JSON.' );
			}

			this._buildGLTF( json, binChunk, path || '' )
				.then( ( result ) => {
					if ( onLoad ) onLoad( result );
				} )
				.catch( ( err ) => {
					if ( onError ) onError( err );
					else console.error( err );
				} );
		} catch ( err ) {
			if ( onError ) onError( err );
			else console.error( err );
		}
	}

	async _buildGLTF( json, binChunk, basePath ) {
		// 1. Resolve Buffers
		const buffers = await Promise.all(
			( json.buffers || [] ).map( async ( bufDef, idx ) => {
				if ( idx === 0 && ! bufDef.uri && binChunk ) {
					return binChunk;
				}
				if ( bufDef.uri ) {
					const url = bufDef.uri.startsWith( 'data:' ) ? bufDef.uri : basePath + bufDef.uri;
					const res = await fetch( url );
					return await res.arrayBuffer();
				}
				return new ArrayBuffer( bufDef.byteLength || 0 );
			} )
		);

		const getBufferViewBytes = ( bvIndex ) => {
			const bv = json.bufferViews[ bvIndex ];
			const buf = buffers[ bv.buffer || 0 ];
			const byteOffset = bv.byteOffset || 0;
			const byteLength = bv.byteLength || 0;
			return { buf, byteOffset, byteLength, byteStride: bv.byteStride || 0 };
		};

		const accessorCache = new Map();
		const getAccessor = ( accIndex ) => {
			if ( accessorCache.has( accIndex ) ) return accessorCache.get( accIndex );
			const acc = json.accessors[ accIndex ];
			const itemSize = TYPE_SIZES[ acc.type ];
			const TypedArray = COMPONENT_TYPES[ acc.componentType ];
			const elemBytes = TypedArray.BYTES_PER_ELEMENT;
			const normalized = Boolean( acc.normalized );

			if ( acc.bufferView === undefined ) {
				const arr = new TypedArray( acc.count * itemSize );
				const attr = new BufferAttribute( arr, itemSize, normalized );
				accessorCache.set( accIndex, attr );
				return attr;
			}

			const bv = getBufferViewBytes( acc.bufferView );
			const accOffset = bv.byteOffset + ( acc.byteOffset || 0 );

			let attr;
			if ( bv.byteStride && bv.byteStride !== itemSize * elemBytes ) {
				const ibSlice = bv.buf.slice( bv.byteOffset, bv.byteOffset + bv.byteLength );
				const ibArray = new TypedArray( ibSlice );
				const strideElems = bv.byteStride / elemBytes;
				const ib = new InterleavedBuffer( ibArray, strideElems );
				attr = new InterleavedBufferAttribute(
					ib,
					itemSize,
					( acc.byteOffset || 0 ) / elemBytes,
					normalized
				);
			} else {
				const byteLen = acc.count * itemSize * elemBytes;
				const slice = bv.buf.slice( accOffset, accOffset + byteLen );
				const arr = new TypedArray( slice );
				attr = new BufferAttribute( arr, itemSize, normalized );
			}
			accessorCache.set( accIndex, attr );
			return attr;
		};

		// 2. Load Images & Textures
		const images = await Promise.all(
			( json.images || [] ).map( async ( imgDef ) => {
				let blobUrl = null;
				let shouldRevoke = false;
				if ( imgDef.bufferView !== undefined ) {
					const bv = getBufferViewBytes( imgDef.bufferView );
					const bytes = new Uint8Array( bv.buf, bv.byteOffset, bv.byteLength );
					const blob = new Blob( [ bytes ], { type: imgDef.mimeType || 'image/png' } );
					blobUrl = URL.createObjectURL( blob );
					shouldRevoke = true;
				} else if ( imgDef.uri ) {
					blobUrl = imgDef.uri.startsWith( 'data:' ) ? imgDef.uri : basePath + imgDef.uri;
				}
				if ( ! blobUrl ) return null;

				try {
					if ( typeof createImageBitmap === 'function' ) {
						const resp = await fetch( blobUrl );
						const b = await resp.blob();
						const bitmap = await createImageBitmap( b, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' } );
						if ( shouldRevoke ) URL.revokeObjectURL( blobUrl );
						return bitmap;
					}
				} catch ( _e ) {
					// Fallback to HTMLImageElement via TextureLoader
				}

				return await new Promise( ( resolve ) => {
					const img = new Image();
					img.crossOrigin = 'anonymous';
					img.onload = () => {
						if ( shouldRevoke ) URL.revokeObjectURL( blobUrl );
						resolve( img );
					};
					img.onerror = () => {
						if ( shouldRevoke ) URL.revokeObjectURL( blobUrl );
						resolve( null );
					};
					img.src = blobUrl;
				} );
			} )
		);

		const buildTexture = ( texInfo, colorSpace ) => {
			if ( ! texInfo || json.textures === undefined ) return null;
			const texDef = json.textures[ texInfo.index ];
			if ( ! texDef ) return null;
			const srcImg = images[ texDef.source ];
			if ( ! srcImg ) return null;

			const tex = new Texture( srcImg );
			tex.flipY = false; // glTF 2.0 standard: UV origin is top-left (flipY = false)
			tex.colorSpace = colorSpace;

			const sampler = texDef.sampler !== undefined && json.samplers ? json.samplers[ texDef.sampler ] : null;
			tex.wrapS = ( sampler && WEBGL_WRAPPINGS[ sampler.wrapS ] ) || RepeatWrapping;
			tex.wrapT = ( sampler && WEBGL_WRAPPINGS[ sampler.wrapT ] ) || RepeatWrapping;
			tex.magFilter = ( sampler && WEBGL_FILTERS[ sampler.magFilter ] ) || LinearFilter;
			tex.minFilter = ( sampler && WEBGL_FILTERS[ sampler.minFilter ] ) || LinearMipmapLinearFilter;
			tex.generateMipmaps = true;
			tex.needsUpdate = true;
			return tex;
		};

		// 3. Build Materials
		const materials = ( json.materials || [] ).map( ( matDef ) => {
			const pbr = matDef.pbrMetallicRoughness || {};
			const ext = matDef.extensions || {};

			if ( ext.KHR_materials_unlit ) {
				const unlitParams = {
					name: matDef.name || '',
					side: matDef.doubleSided ? DoubleSide : FrontSide,
				};
				if ( pbr.baseColorFactor ) {
					unlitParams.color = new Color().fromArray( pbr.baseColorFactor, 0 );
					if ( pbr.baseColorFactor[ 3 ] < 1.0 ) unlitParams.opacity = pbr.baseColorFactor[ 3 ];
				}
				if ( pbr.baseColorTexture ) {
					unlitParams.map = buildTexture( pbr.baseColorTexture, SRGBColorSpace );
				}
				if ( matDef.alphaMode === 'BLEND' ) {
					unlitParams.transparent = true;
					unlitParams.depthWrite = false;
				} else if ( matDef.alphaMode === 'MASK' ) {
					unlitParams.alphaTest = matDef.alphaCutoff !== undefined ? matDef.alphaCutoff : 0.5;
				}
				return new MeshBasicMaterial( unlitParams );
			}

			const usePhysical = Boolean( ext.KHR_materials_transmission || ext.KHR_materials_ior || ext.KHR_materials_clearcoat );
			const MatClass = usePhysical ? MeshPhysicalMaterial : MeshStandardMaterial;

			const params = {
				name: matDef.name || '',
				metalness: pbr.metallicFactor !== undefined ? pbr.metallicFactor : ( pbr.metallicRoughnessTexture ? 1.0 : 0.0 ),
				roughness: pbr.roughnessFactor !== undefined ? pbr.roughnessFactor : 1.0,
				side: matDef.doubleSided ? DoubleSide : FrontSide,
			};

			if ( pbr.baseColorFactor ) {
				params.color = new Color().setRGB(
					pbr.baseColorFactor[ 0 ],
					pbr.baseColorFactor[ 1 ],
					pbr.baseColorFactor[ 2 ],
					LinearSRGBColorSpace
				);
				if ( pbr.baseColorFactor[ 3 ] !== undefined ) {
					params.opacity = pbr.baseColorFactor[ 3 ];
				}
			}

			if ( pbr.baseColorTexture ) {
				params.map = buildTexture( pbr.baseColorTexture, SRGBColorSpace );
			}

			if ( pbr.metallicRoughnessTexture ) {
				const mrTex = buildTexture( pbr.metallicRoughnessTexture, NoColorSpace );
				params.roughnessMap = mrTex;
				params.metalnessMap = mrTex;
			}

			if ( matDef.normalTexture ) {
				params.normalMap = buildTexture( matDef.normalTexture, NoColorSpace );
				const s = matDef.normalTexture.scale !== undefined ? matDef.normalTexture.scale : 1.0;
				params.normalScale = new Vector2( s, s );
			}

			if ( matDef.occlusionTexture ) {
				params.aoMap = buildTexture( matDef.occlusionTexture, NoColorSpace );
				if ( matDef.occlusionTexture.strength !== undefined ) {
					params.aoMapIntensity = matDef.occlusionTexture.strength;
				}
			}

			if ( matDef.emissiveFactor ) {
				params.emissive = new Color().setRGB(
					matDef.emissiveFactor[ 0 ],
					matDef.emissiveFactor[ 1 ],
					matDef.emissiveFactor[ 2 ],
					LinearSRGBColorSpace
				);
			}

			if ( matDef.emissiveTexture ) {
				params.emissiveMap = buildTexture( matDef.emissiveTexture, SRGBColorSpace );
				if ( ! params.emissive ) params.emissive = new Color( 0xffffff );
			}

			if ( ext.KHR_materials_emissive_strength && ext.KHR_materials_emissive_strength.emissiveStrength !== undefined ) {
				params.emissiveIntensity = ext.KHR_materials_emissive_strength.emissiveStrength;
			}

			if ( usePhysical ) {
				if ( ext.KHR_materials_transmission && ext.KHR_materials_transmission.transmissionFactor !== undefined ) {
					params.transmission = ext.KHR_materials_transmission.transmissionFactor;
				}
				if ( ext.KHR_materials_ior && ext.KHR_materials_ior.ior !== undefined ) {
					params.ior = ext.KHR_materials_ior.ior;
				}
			}

			if ( matDef.alphaMode === 'BLEND' ) {
				params.transparent = true;
				params.depthWrite = false;
			} else if ( matDef.alphaMode === 'MASK' ) {
				params.alphaTest = matDef.alphaCutoff !== undefined ? matDef.alphaCutoff : 0.5;
			}

			return new MatClass( params );
		} );

		const defaultMaterial = new MeshStandardMaterial( { color: 0xffffff, roughness: 0.6, metalness: 0.0 } );

		// 4. Build Meshes
		const buildMeshObject = ( meshDef ) => {
			const prims = meshDef.primitives || [];
			const builtMeshes = prims.map( ( prim, pIdx ) => {
				const geo = new BufferGeometry();
				const attrs = prim.attributes || {};
				for ( const [ gltfAttr, accIdx ] of Object.entries( attrs ) ) {
					const threeAttr = ATTRIBUTE_MAP[ gltfAttr ] || gltfAttr.toLowerCase();
					const bufAttr = getAccessor( accIdx );
					geo.setAttribute( threeAttr, bufAttr );
					if ( threeAttr === 'color' ) {
						// Vertex colors
					}
				}
				if ( prim.indices !== undefined ) {
					geo.setIndex( getAccessor( prim.indices ) );
				}
				if ( ! geo.hasAttribute( 'normal' ) ) {
					geo.computeVertexNormals();
				}
				geo.computeBoundingBox();
				geo.computeBoundingSphere();

				let mat = prim.material !== undefined ? materials[ prim.material ] || defaultMaterial : defaultMaterial;
				if ( geo.hasAttribute( 'color' ) && ! mat.vertexColors ) {
					mat = mat.clone();
					mat.vertexColors = true;
				}

				const mesh = new Mesh( geo, mat );
				mesh.name = prims.length === 1 ? ( meshDef.name || `Mesh_${pIdx}` ) : `${meshDef.name || 'Mesh'}_${pIdx}`;
				mesh.castShadow = true;
				mesh.receiveShadow = true;
				return mesh;
			} );

			if ( builtMeshes.length === 1 ) return builtMeshes[ 0 ];
			const grp = new Group();
			grp.name = meshDef.name || 'MeshGroup';
			for ( const m of builtMeshes ) grp.add( m );
			return grp;
		};

		// 5. Build Nodes & Scene Graph
		const buildNode = ( nodeIdx ) => {
			const nodeDef = json.nodes[ nodeIdx ];
			let obj;
			if ( nodeDef.mesh !== undefined ) {
				obj = buildMeshObject( json.meshes[ nodeDef.mesh ] );
			} else {
				obj = new Group();
			}
			if ( nodeDef.name ) obj.name = nodeDef.name;
			if ( nodeDef.extras ) obj.userData = Object.assign( {}, nodeDef.extras );

			if ( nodeDef.matrix ) {
				const m = new Matrix4().fromArray( nodeDef.matrix );
				m.decompose( obj.position, obj.quaternion, obj.scale );
			} else {
				if ( nodeDef.translation ) obj.position.fromArray( nodeDef.translation );
				if ( nodeDef.rotation ) obj.quaternion.fromArray( nodeDef.rotation );
				if ( nodeDef.scale ) obj.scale.fromArray( nodeDef.scale );
			}

			if ( nodeDef.children ) {
				for ( const childIdx of nodeDef.children ) {
					obj.add( buildNode( childIdx ) );
				}
			}
			return obj;
		};

		const scenes = ( json.scenes || [] ).map( ( sceneDef, sIdx ) => {
			const sceneRoot = new Group();
			sceneRoot.name = sceneDef.name || `Scene_${sIdx}`;
			for ( const rootNodeIdx of sceneDef.nodes || [] ) {
				sceneRoot.add( buildNode( rootNodeIdx ) );
			}
			return sceneRoot;
		} );

		const activeScene = scenes[ json.scene || 0 ] || scenes[ 0 ] || new Group();

		return {
			scene: activeScene,
			scenes,
			animations: [],
			cameras: [],
			asset: json.asset || {},
			userData: json.extras || {},
		};
	}
}
